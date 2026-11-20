# Backend neurali di trascrizione e confronto tra frasi: note di verifica

**Data:** 2026-11-20. Issue #12, #13, #15. **Codice:** `src/akoe/transcribe/`.

Ogni backend registra in `ModelProvenance` (`src/akoe/neural.py`):
- identificativo del modello e revisione (commit Hugging Face);
- SHA-256 di ogni file caricato, adattatori compresi;
- dispositivo.

`stage_log_fields()` riempie `StageLog.model_version`, `model_sha256` e `device`.

## Modelli e licenze

| Uso | Modello (revisione fissata) | Licenza (campo `license` della model card, verificato) |
|---|---|---|
| Trascrizione, seconda famiglia | `openai/whisper-tiny` @ `169d4a43`, `openai/whisper-base` @ `e37978b9` | apache-2.0 |
| Allineamento a caratteri, predefinito (issue #58) | `jonatasgrosman/wav2vec2-xls-r-1b-italian` @ `02d4be26` (XLS-R 1B, Common Voice 8 it) | apache-2.0 |
| Frasi candidate, predefinito (issue #58) | omniASR CTC 1B v2 (EDGE), 3B v2 (WORKSTATION/CLOUD, quando lo SHA-256 è fissato): checkpoint ufficiali Meta, `src/akoe/transcribe/omniasr_ctc.py` | apache-2.0 (repository `facebookresearch/omnilingual-asr`) |
| Punteggio CTC, precedente | `jonatasgrosman/wav2vec2-large-xlsr-53-italian` @ `dab04a3e` (XLS-R 53, Common Voice 6.1 it) | apache-2.0 |
| Punteggio CTC, solo test | `facebook/wav2vec2-base-10k-voxpopuli-ft-it` @ `653b3c78` | cc-by-nc-4.0 (non commerciale) |
| Trascrizione principale | `nvidia/parakeet-tdt-0.6b-v3` @ `541d1f99` | cc-by-4.0 |

**Altri modelli CTC italiani visti e non scelti:**
- `dbdmg/wav2vec2-xls-r-300m-italian` (apache-2.0, poco usato);
- `facebook/mms-1b-all` (cc-by-nc-4.0, 1 miliardo di parametri);
- `MahmoudAshraf/mms-300m-1130-forced-aligner` (cc-by-nc-4.0);
- `nvidia/stt_it_fastconformer_hybrid_large_pc` (cc-by-4.0, ibrido RNNT+CTC, richiede NeMo).

Non verificate le condizioni dei dati di addestramento (Common Voice, Granary).

## omniASR CTC v2 per le frasi candidate (`omniasr_ctc.py`, issue #58)

- **Pesi:** checkpoint ufficiali Meta (`dl.fbaipublicfiles.com/mms/omniASR-CTC-{1B,3B}-v2.pt`, citati dalle model card di `facebookresearch/omnilingual-asr` @ `81f51e22`, Apache-2.0) e tokenizer `omniASR_tokenizer_written_v2.model`.
  - SHA-256 fissati: tokenizer `8aa11a10…0b1e`; 1B `354f9817…919c`.
  - Il 3B (12,3 GB) non è stato scaricato: finché il suo hash non è fissato viene rifiutato, e anche WORKSTATION e CLOUD usano il 1B.
- **Senza fairseq2:**
  - il checkpoint contiene solo tensori: si legge con `torch.load(weights_only=True)`, mappato dal file;
  - i nomi fairseq2 corrispondono uno a uno a `Wav2Vec2ForCTC` di transformers (architettura XLS-R con layer norm prima dei blocchi), con caricamento rigoroso;
  - ingresso normalizzato a media zero e varianza uno, al massimo 40 s, blank = indice 0.
- **Vocabolario di 10288 caratteri in forma scritta** (maiuscole, punteggiatura, cifre).
  - Prima dell'allineamento forzato (Viterbi, un token per carattere), maiuscole e minuscole della stessa lettera vengono sommate. Blank, punteggiatura (tranne l'apostrofo) e simboli speciali diventano un unico blank.
  - Così il punteggio non dipende da come l'operatore scrive maiuscole e punteggiatura.
- **Prova su audio reale (2026-11-20, CPU condivisa):** 13 clip italiane (`example-it.wav` di Common Voice e 12 clip MInDS-14 al telefono a 8 kHz).
  - **Trascrizione greedy:** CER 0 su Common Voice, CER medio 7,2%. Diversi "errori" sono imprecisioni dei testi di riferimento di MInDS-14. La conversione dei pesi quindi funziona.
  - **Frase vera contro tre frasi alterate** (parole scambiate, tolte o sostituite): la vera è prima in 11 casi su 13. Nei 2 casi restanti l'alterazione tocca una parola che il modello sente diversamente dal riferimento ("voglio", "mago"), probabile imprecisione del riferimento; lo scarto è minimo (0,02 e 0,0003).
  - **Costi:** circa 6 s per clip di 8-18 s; picco di memoria residente 4,6 GB, con i pesi mappati dal file.

## Whisper (`whisper_backend.py`)

- **Alternative.** Le N migliori ipotesi vengono dal beam search, più eventuali campioni con temperatura e soglia epsilon (seme registrato).
  - Tutti i candidati restano, con l'origine:
    - `whisper-tiny/generic` o `whisper-tiny/adapted:<id>` per il beam;
    - lo stesso più `/sampling` per i candidati trovati solo dal campionamento.
  - `mbr_select` indica il candidato più centrale senza scartare gli altri.
- **Punteggio unico per tutti i candidati:** log-probabilità media per token del testo, forzato nel decoder sullo stesso audio. Gli `sequences_scores` del beam usano un'altra normalizzazione e non sono confrontabili con i campioni.
- **Problema trovato in transformers 5.23.** `WhisperForConditionalGeneration.generate` con `num_return_sequences` ripete l'ingresso ed esegue ricerche indipendenti: restituisce N copie della stessa ipotesi. Il backend usa il `generate` generico con il prompt del decoder esplicito e i token di timestamp soppressi. Un test lento controlla la regressione.
- **Adattatori LoRA:** cartella salvata con peft `save_pretrained`, caricata con `PeftModel.from_pretrained`. Si rifiuta un adattatore addestrato su un modello di base diverso.
- **Tempi e confidenza delle parole:**
  - tempi da cross-attention e DTW sulle `alignment_heads`, solo per l'ipotesi migliore;
  - confidenza = probabilità congiunta dei token della parola, grezza e non calibrata.
- **Limiti:**
  - la prima parola tende ad assorbire il silenzio iniziale;
  - i punteggi dipendono da maiuscole e punteggiatura;
  - finestre di al massimo 30 s (i tratti più lunghi vengono divisi).

## Frasi candidate (`ctc_scorer.py`, `candidates.py`)

- **Allineamento.** Allineamento forzato CTC con Viterbi sul traliccio, implementato in numpy.
  - `torchaudio.functional.forced_align` era annunciato come deprecato dalla 2.8 e in rimozione nella 2.9.
  - La 2.11.1 installata lo contiene ancora e funziona: i test lo usano solo come controllo (punteggi identici).
- **Punteggio** (`score`): log-probabilità per frame del percorso forzato meno quella del percorso libero migliore.
  - Vale 0 se la frase coincide con ciò che il modello decodificherebbe da solo, sempre ≤ 0.
  - Sono riportati anche il valore assoluto per frame e quello per token, e l'allineamento per parola.
- **Normalizzazione del testo:** minuscole; apostrofi e trattini come separatori se l'alfabeto non li ha; lettere accentate mancanti sostituite dalla base. Cifre e simboli vengono scartati con avviso: scrivere i numeri in lettere.
- **`compare_candidates`:**
  - riporta **separatamente** il punteggio CTC e quello del decoder Whisper, senza mai unirli;
  - restituisce "nessuna compatibile" solo se tutti i metodi con soglia concordano, e "disaccordo" se divergono;
  - le soglie vengono dall'autovalutazione (M5).
- **Problema trovato in un modello.** In `facebook/wav2vec2-base-10k-voxpopuli-ft-it` il `pad_token_id` del config (1) non coincide con `<pad>` del vocabolario (0). Il blank si prende dal vocabolario.

## Parakeet-TDT e NeMo (`parakeet_backend.py`)

**Fattibilità di NeMo con Python 3.13: sì.** Prova in un ambiente usa e getta in `/tmp`, mai nell'ambiente principale:

```sh
python3.13 -m venv /percorso/nemo-venv
/percorso/nemo-venv/bin/pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
/percorso/nemo-venv/bin/pip install "nemo_toolkit[asr]" --extra-index-url https://download.pytorch.org/whl/cpu
/percorso/nemo-venv/bin/pip install -e /percorso/akoe   # questo pacchetto
```

- **Versioni installate:** `nemo_toolkit` 3.0.0, torch 2.15.0+cpu, circa 2,2 GB su disco. Import riuscito.
- **Senza NeMo:** il backend solleva `BackendUnavailable` (sottoclasse di `NotImplementedError`, registrata come stadio saltato) con le istruzioni in italiano.
- **Confidenza:** `confidence_cfg` con entropia di Tsallis, α = 0,33, normalizzazione esponenziale, aggregazione per parola "min", blank escluso.
- **N migliori:** strategia `malsd_batch`, quella raccomandata da NeMo. L'ipotesi greedy porta tempi e confidenza; quelle del beam hanno origine `/beam` e il punteggio del beam.
- **Verifica del codice:** con `nvidia/stt_it_fastconformer_hybrid_large_pc` (115 M parametri, italiano, cc-by-4.0) su CPU funzionano tempi per parola, confidenza entropica e N migliori.
  - **Parakeet-TDT 0.6B v3 non è stato eseguito:** 2,5 GB di pesi in float32, e il caricamento di NeMo (istanzia il modello e poi `torch.load` senza mmap) porterebbe il picco a circa 5 GB, oltre il limite di 3 GB di questa macchina condivisa.
  - **Da provare** su una macchina con più RAM.
  - **Alternativa senza NeMo:** transformers 5.23 contiene `ParakeetForTDT` e il repository ha `model.safetensors`, ma la confidenza entropica andrebbe reimplementata.
- **Testa CTC di Parakeet-TDT v3: assente.**
  - Il `.nemo` dichiara `EncDecRNNTBPEModel` (solo TDT).
  - Il config conserva un blocco `aux_ctc` residuo (`num_classes: -1`, vocabolario vuoto).
  - Il checkpoint contiene solo i pesi `preprocessor`, `encoder`, `decoder`, `joint`, nessuna chiave `ctc`: chiavi lette scaricando solo l'indice del checkpoint.
  - NeMo Forced Aligner richiede modelli CTC o ibridi, quindi l'allineamento forzato resta al modello CTC separato.
  - Un'alternativa NeMo con testa CTC per l'italiano è il modello ibrido sopra.
