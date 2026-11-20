# Backend neurali di trascrizione e confronto tra frasi: note di verifica

**Data:** 2026-11-20. Issue #12, #13, #15, #54, #55, #56. **Codice:** `src/akoe/transcribe/`.

Ogni backend registra in `ModelProvenance` (`src/akoe/neural.py`):
- identificativo del modello e revisione (commit Hugging Face);
- SHA-256 di ogni file caricato, adattatori compresi;
- dispositivo.

`stage_log_fields()` riempie `StageLog.model_version`, `model_sha256` e `device`.

## Modelli e licenze

| Uso | Modello (revisione fissata) | Licenza (campo `license` della model card, verificato) |
|---|---|---|
| Trascrizione, seconda famiglia | `openai/whisper-tiny` @ `169d4a43`, `openai/whisper-base` @ `e37978b9` | apache-2.0 |
| Punteggio CTC, predefinito | `jonatasgrosman/wav2vec2-large-xlsr-53-italian` @ `dab04a3e` (XLS-R 53, Common Voice 6.1 it) | apache-2.0 |
| Punteggio CTC, solo test | `facebook/wav2vec2-base-10k-voxpopuli-ft-it` @ `653b3c78` | cc-by-nc-4.0 (non commerciale) |
| Trascrizione principale | `nvidia/parakeet-tdt-0.6b-v3` @ `541d1f99` | cc-by-4.0 |
| Trascrizione, tutti i livelli | `CohereLabs/cohere-transcribe-03-2026` @ `d552dadc` (accesso controllato) | apache-2.0 |
| Trascrizione, tutti i livelli | `Qwen/Qwen3-ASR-1.7B-hf` @ `bcd2b5b7` | apache-2.0 |
| Tempi delle parole di Qwen3-ASR | `Qwen/Qwen3-ForcedAligner-0.6B-hf` @ `c07281df` | apache-2.0 |
| Trascrizione, solo CLOUD | `nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B` @ `c1766764` | openmdw-1.1 (campo `license_name`) |

SHA-256 dei pesi pubblicati dal hub (metadati LFS), controllati al caricamento: un file diverso viene rifiutato.

| File | SHA-256 |
|---|---|
| Cohere `model.safetensors` (4,13 GB, bf16) | `987bd3e141c7bfdb5a78f5db11397ee7737308357e6cc0a3f36a4979b158137a` |
| Qwen3-ASR-1.7B-hf `model.safetensors` (4,08 GB, bf16) | `2db53c7d81bd9b8cbc6a074e89be2c968a0d373fb4ee68bb1b1e14f7042dfee1` |
| Qwen3-ForcedAligner-0.6B-hf `model.safetensors` (1,84 GB, bf16) | `00568245ceca5af1991d28562a75fe1ddc9bfeb041c27fda66947ea05c47fb86` |
| Parakeet `parakeet-tdt-0.6b-v3.nemo` (2,5 GB) | `3cbdc85877e668ca7b82d0d56770eb1fac76691f55d6b97545e8d61ca588d10d` (calcolato) |

**Altri modelli CTC italiani visti e non scelti:**
- `dbdmg/wav2vec2-xls-r-300m-italian` (apache-2.0, poco usato);
- `facebook/mms-1b-all` (cc-by-nc-4.0, 1 miliardo di parametri);
- `MahmoudAshraf/mms-300m-1130-forced-aligner` (cc-by-nc-4.0);
- `nvidia/stt_it_fastconformer_hybrid_large_pc` (cc-by-4.0, ibrido RNNT+CTC, richiede NeMo).

Non verificate le condizioni dei dati di addestramento (Common Voice, Granary).

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
- **Parakeet-TDT 0.6B v3 eseguito il 2026-11-20** (ambiente NeMo separato in `/tmp/nemo_venv`: Python 3.13.5, NeMo 3.0.0, torch 2.15.0+cpu; `ParakeetTDTBackend` del repository, `n_best=3`, CPU, 6 thread):
  - **picco di RAM 3,95 GB** (RSS del processo; 3,6 GB dopo il caricamento);
  - caricamento 853 s, ma con la macchina in swap per altri processi: non è un tempo rappresentativo;
  - trascrizione: 3,5 s per 8,4 s di audio telefonico, 4,9 s per 16,8 s (la prima chiamata, 244 s, include il riscaldamento sotto swap).
  - **Risultati** (riferimento → ipotesi migliore):
    - Common Voice, letto, 4,3 s: "Buongiorno a tutti e benvenuti a bordo" → "Buongiorno a tutti e benvenuti a bordo." (esatto);
    - MINDS-14 `minds_it_01`, telefono 8 kHz, spontaneo: "Buongiorno Vorrei depositare denaro sul mio conto corrente Come devo fare per cortesia" → "Buongiorno e senta, vorrei depositare denaro sul mio conto corrente. Come devo fare, per cortesia?" ("e senta" probabilmente presente nell'audio e omesso dal riferimento; confidenza 0,01 e 0,002);
    - MINDS-14 `minds_it_03`, 16,8 s: unica differenza di contenuto "Voglio" → "Volevo", "un altro conto" → "in un altro conto"; le ripetizioni "come come", "se se" sono state ripulite dal modello.
  - **La confidenza entropica segnala i punti critici:** "come" della ripetizione 0,00, "in" aggiunto 0,00, "bancomat." 0,00, mentre le parole chiare stanno tra 0,6 e 0,9.
  - **Attenzione per il lavoro forense:** Parakeet non è verbatim (toglie ripetizioni ed esitazioni); le altre famiglie dell'insieme servono anche a questo.
  - **Punteggi:** l'ipotesi greedy ha un punteggio su un'altra scala (−498, −895) rispetto a quelle del beam (−3,4, −5,6): l'origine `/beam` le distingue e non vanno confrontati tra loro.
  - **Alternativa senza NeMo:** transformers 5.23 contiene `ParakeetForTDT` e il repository ha `model.safetensors`, ma la confidenza entropica andrebbe reimplementata.
- **Testa CTC di Parakeet-TDT v3: assente.**
  - Il `.nemo` dichiara `EncDecRNNTBPEModel` (solo TDT).
  - Il config conserva un blocco `aux_ctc` residuo (`num_classes: -1`, vocabolario vuoto).
  - Il checkpoint contiene solo i pesi `preprocessor`, `encoder`, `decoder`, `joint`, nessuna chiave `ctc`: chiavi lette scaricando solo l'indice del checkpoint.
  - NeMo Forced Aligner richiede modelli CTC o ibridi, quindi l'allineamento forzato resta al modello CTC separato.
  - Un'alternativa NeMo con testa CTC per l'italiano è il modello ibrido sopra.

## Insieme di trascrittori (`engine.py`, `DefaultBackends`)

Più famiglie insieme danno alternative e un controllo incrociato delle allucinazioni (`docs/catalogo_modelli.md`).

- **Insieme predefinito** (`default_asr_ensemble`), nell'ordine:
  - EDGE e WORKSTATION: Cohere Transcribe, Qwen3-ASR, Parakeet (solo se NeMo è installato), Whisper generico, più Whisper adattato se c'è un adattatore verificato;
  - CLOUD: Nemotron 3.5 Transcribe in testa, poi gli stessi.
  - La prima fonte che gira fa da scheletro dell'allineamento parola per parola (`combine.py`).
- **Configurazione:**
  - `DefaultBackends(asr_ensemble="cohere,qwen3,whisper")` oppure la variabile `AKOE_ASR_ENSEMBLE` (stessi nomi, separati da virgole);
  - nomi validi: `nemotron`, `cohere`, `qwen3`, `parakeet`, `whisper`;
  - `engine.transcribe(..., asr_systems=["cohere", "qwen3"])` tiene solo le etichette indicate (`cohere`, `qwen3`, `parakeet`, `generic`, `adapted`, `nemotron`); le etichette assenti vengono annotate nel registro.
- **Un trascrittore che manca non ferma gli altri:** modello non scaricabile, accesso controllato, NeMo o GPU assenti diventano uno stadio "saltato" con il motivo in italiano.
- **Memoria (EDGE, 14 GB condivisi):** con `low_memory` (predefinito) ogni modello viene liberato dopo il suo turno in `transcribe`, quindi il picco è quello del modello più grande e non la somma. Cohere e Qwen3, con `low_memory`, liberano anche i pesi del riconoscitore prima di caricare l'allineatore.
- **Limite noto:** `self_assess` tiene caricati tutti i trascrittori dell'insieme per tutta la durata; su EDGE conviene ridurre l'insieme (`AKOE_ASR_ENSEMBLE`) per l'autovalutazione.

## Struttura comune di Cohere e Qwen3 (`hf_asr.py`)

Stesse convenzioni di Whisper:
- **alternative** dal beam search (origine `<modello>/generic`) e, se configurato, dal campionamento con seme registrato (origine `.../sampling`);
- **un solo punteggio per tutti i candidati:** log-probabilità media per token, forzata nel decoder sullo stesso audio (token di fine compreso), così beam e campioni sono confrontabili;
- **confidenza per parola:** prodotto delle probabilità dei token della parola nell'ipotesi migliore, grezza e non calibrata (`word_timing.py`, raggruppamento dei token indipendente dal tokenizer);
- **tempi per parola** da un allineatore esterno; una parola che l'allineatore non sa collocare (solo cifre o punteggiatura) riceve l'intervallo tra le parole vicine, mai un tempo inventato;
- **pesi in bfloat16** per default (il formato pubblicato): su una CPU senza bfloat16 nativo (solo AVX2) è più lento del float32, ma è l'unica scelta che sta nella RAM di EDGE;
- **il codice remoto dei repository non viene mai eseguito** (`trust_remote_code` assente): entrambi i modelli sono nativi in transformers 5.23.

## Cohere Transcribe 03-2026 (`cohere_backend.py`, #54)

- **Modello:** 2B parametri, encoder Conformer (48 strati) e decoder Transformer leggero (8 strati); 14 lingue, italiano compreso. Classe nativa `CohereAsrForConditionalGeneration` (transformers ≥ 5.4).
- **Esecuzione:** prompt del decoder dal processore (`<|it|>`, punteggiatura attiva, niente ITN, timestamp o diarizzazione). L'encoder gira una volta per segmento; beam, campionamento e punteggio forzato ne riusano l'uscita.
- **Problema trovato in transformers 5.23:** `generate` espande sul posto l'uscita dell'encoder per i beam; il backend gli passa ogni volta un contenitore nuovo.
- **Tempi per parola:** il modello non ne produce (model card: "does not feature" timestamp né diarizzazione). Vengono dall'allineatore CTC italiano del repository (`CTCTranscriptAligner.word_times`, nuovo metodo che usa lo stesso allineamento forzato di `align`), con modello e SHA-256 registrati nella provenienza (`word_timer`).
- **Riproducibilità:** il dither dell'estrattore di caratteristiche (1e-5) ha un seme fissato dalla lunghezza del tratto in transformers: risultati ripetibili. Il valore viene registrato.
- **Finestre:** tratti oltre 30 s divisi; il processore li dividerebbe da sé oltre 30 s e incollerebbe i testi, nascondendo i confini.
- **Accesso controllato:** il repository richiede di accettare le condizioni (approvazione automatica). Senza un token accettato il backend solleva `BackendUnavailable` con le istruzioni. In alternativa si può indicare come modello una cartella locale con gli stessi file: lo SHA-256 di `model.safetensors` viene controllato comunque.
- **Memoria:** 4,1 GB in bfloat16; in float32 servirebbero 8,3 GB.
- **Verifica:** i test veloci usano un modello minuscolo con le classi vere di transformers (N migliori, punteggi forzati, campionamento riproducibile, bfloat16, rilascio della memoria, controllo degli hash).
  - **Il modello vero non è stato eseguito:** su questa macchina non c'è un token Hugging Face con le condizioni accettate. Serve un'azione del proprietario (vedi sopra); il test lento `tests/test_cohere_backend.py::test_real_model_italian` lo esegue appena i pesi sono disponibili (anche con `AKOE_TEST_COHERE_DIR`).

## Nemotron 3.5 Transcribe (`nemotron_backend.py`, #56) — solo livello CLOUD

- **Modello:** `nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B` (ottobre 2026), licenza OpenMDW-1.1. Modello linguistico con ingresso vocale (SALM): encoder ASR da 600M, encoder di diarizzazione, teste CTC per i tempi, Nemotron 3.5 Lightning 30B-A3B (circa 4,3B parametri attivi). Il miglior italiano aperto: FLEURS-it 1,41, media 3,61 (model card).
- **Requisiti:** un file `model.safetensors` da 67,5 GB in BF16; la model card indica una H100 da 80 GB. Esistono varianti FP8 e NVFP4 per altre GPU.
- **Mai eseguito su questa macchina** (CPU, 14 GB) e mai scaricato. Il backend rifiuta il caricamento con un messaggio in italiano (`BackendUnavailable`, registrato come stadio saltato) se:
  - manca NeMo Speech (`nemo.collections.speechlm2`);
  - non c'è una GPU CUDA con almeno 78 GB di memoria (parametro `min_gpu_memory_gb`).
- **Installazione sulla VM CLOUD** (dalla model card, sezione "Environment Setup"; ambiente separato da quello di sviluppo):

```sh
git clone https://github.com/NVIDIA-NeMo/Speech.git nemo-speech
cd nemo-speech
uv venv --python 3.13 .venv
UV_PROJECT_ENVIRONMENT=$PWD/.venv uv sync --extra asr --locked --no-install-project
uv pip install --python .venv/bin/python --no-deps -e .
uv pip install --python .venv/bin/python -e /percorso/akoe     # questo pacchetto
.venv/bin/hf download nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B \
    --revision c17667641642c4265ff2b7037dd101b7d4567634 --local-dir /dati/nemotron-3.5-transcribe
```

  - Poi `NemotronTranscribeBackend("/dati/nemotron-3.5-transcribe", device="cuda")`, oppure l'insieme predefinito del livello CLOUD.
  - Per il servizio con vLLM (la via raccomandata dalla model card per la velocità) servono anche `--extra vllm`, vLLM 0.32.0 e il plugin `nemo_speechlm`; il server deve girare sulla VM dell'utente, mai su un servizio di terzi.
- **Uso in processo:** API SALM di NeMo Speech (`SALMAutomodel.from_pretrained`, `generate(prompts=..., audios=..., audio_lens=..., generation_config=...)`), prompt verbatim della model card con la lingua fissata ("Language: Italian"), ragionamento disattivato.
  - **Da confermare sulla VM:** la classe da caricare e il formato del prompt con la versione di NeMo Speech installata; questo percorso non è stato eseguito.
- **Uscite:** N migliori dal beam search, senza punteggio (l'API restituisce solo i token); tempi per parola dall'allineatore CTC italiano del repository (le teste CTC del modello sono esposte solo dal middleware di vLLM); nessuna confidenza per parola.

## Qwen3-ASR-1.7B e Qwen3-ForcedAligner (`qwen3_backend.py`, #55)

- **Modelli:** i repository "-hf" pubblicati da Qwen nel formato nativo di transformers (`Qwen3ASRForConditionalGeneration`, `Qwen3ASRForTokenClassification`, transformers 5.23); i repository originali sono per il pacchetto `qwen-asr`, che qui non serve. Nessun pacchetto aggiuntivo installato.
- **Lingua forzata** come nell'implementazione di riferimento: il turno dell'assistente viene precompilato con "language Italian<asr_text>", quindi il modello scrive solo la trascrizione e non può passare a un'altra lingua su audio poco chiaro.
- **Efficienza:** l'encoder audio gira una volta per segmento; le embedding del prompt (testo + audio) vengono riusate da beam search, campionamento e punteggio forzato (per il punteggio si tengono solo i logit che predicono il testo).
- **Nessuna correzione del testo:** il correttore di ripetizioni del pacchetto di riferimento non viene applicato; le ripetizioni restano visibili e passano alla guardia contro le allucinazioni.
- **Tempi per parola:** Qwen3-ForcedAligner sull'ipotesi migliore (classi da 80 ms, correzione di monotonia del riferimento). Le parole fatte solo di punteggiatura ricevono l'intervallo tra le vicine. Con `low_memory` i pesi del riconoscitore (3,9 GB) vengono liberati prima di caricare l'allineatore (1,7 GB).
- **Verifica veloce:** modelli minuscoli con le classi vere di transformers (riconoscitore e allineatore, stesso formato di prompt dei repository pubblicati).
- **Eseguito il 2026-11-20 su questa macchina** (ambiente principale, transformers 5.23, torch 2.15 CPU, bfloat16, 6 thread, `low_memory`, `n_best=3`, 4 beam):
  - **picco di RAM 4,36 GB** (RSS campionato; il riconoscitore e l'allineatore non sono mai in memoria insieme);
  - tempi per chiamata, compreso il ricaricamento dei pesi a ogni chiamata (`low_memory`) e con la macchina condivisa sotto carico: 197 s per 4,3 s, 132 s per 8,4 s, 245 s per 16,8 s. Nel flusso dell'engine i pesi si caricano una volta per reperto, non per segmento;
  - su questa CPU (solo AVX2) il bfloat16 è 2,5 volte più veloce del float32 nella decodifica token per token (limitata dalla memoria) e 2,3 volte più lento nel prefill: resta il predefinito.
  - **Risultati** (riferimento → ipotesi migliore; stesse clip di Parakeet):
    - Common Voice, 4,3 s: esatto ("Buongiorno a tutti e benvenuti a bordo."); alternative "Buon giorno…", "buongiorno…"; confidenze 0,93–1,00;
    - `minds_it_01`: "Buongiorno. Vorrei depositare denaro sul mio conto corrente. Come devo fare, per cortesia?" (esatto rispetto al riferimento); la terza alternativa contiene "E senta", come l'ipotesi di Parakeet;
    - `minds_it_03`: **verbatim**, conserva le ripetizioni "come come" e "se se" del riferimento (Parakeet le toglie); errori: "Voglio" → "Volevo" (come Parakeet), "un altro conto" → "dal mio conto" (la terza alternativa ha "da un altro conto").
  - **Tempi dell'allineatore:** plausibili sul parlato letto; su `minds_it_01` sono spostati dove l'ipotesi omette parole presenti nell'audio ("e senta"), e qualche parola breve riceve durata nulla. I tempi seguono il testo allineato: vanno letti insieme alle alternative.
