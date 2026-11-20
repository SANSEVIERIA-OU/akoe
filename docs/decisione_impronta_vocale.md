# Decisione: modello per l'impronta vocale (issue #9, #57)

**Data:** 2026-11-20, seconda versione (issue #57). **Stato:** adottata nel codice; da confermare per caso con l'autovalutazione per condizione (M5).

L'impronta vocale in Akoe produce un **punteggio**, mai un'identificazione. Serve a smistare, selezionare e avvisare ("possibile altro parlante"). Non serve per la sintesi della voce (`MASTER_PLAN.md` §3.7).

## Scelta

Criterio del proprietario (2026-11-20): il migliore disponibile al mondo, con il solo vincolo di girare su macchine controllate (`docs/catalogo_modelli.md`).

1. **Predefinito su tutti i livelli: ReDimNet2-B6** (`PalabraAI/redimnet2`), variante `b6-vb2+vox2_v0-lm`, implementato come `ReDimNet2Voiceprint`.
2. **Massima accuratezza: w2v-BERT 2.0 + adattatori + MFA** (`ZXHY-82/w2v-BERT-2.0_SV`, pesi `zl389/w2v-bert-2.0_SV`), implementato come `W2VBertSVVoiceprint` (`--voiceprint w2v-bert-sv`). Licenza non commerciale: vedi sotto.
3. **Confronto e ripiego:** ReDimNet-B2/B6 v1 (`ReDimNetVoiceprint`) ed ECAPA-TDNN di SpeechBrain (`EcapaVoiceprint`).
4. **Punteggi:** coseno dopo sottrazione della media, **AS-norm** e **QMF** (ricetta WeSpeaker), poi la calibrazione leave-one-recording-out già esistente (`speaker/score_norm.py`, `speaker/profile.py`).

Un profilo costruito con un modello non si confronta con un altro: cambiando il predefinito, i profili esistenti (ReDimNet-B2) vanno ricostruiti; finché non lo sono, il motore lo segnala e non calcola i punteggi.

## Evidenze (`MASTER_PLAN.md` §3.11 e fonti dei modelli)

- **ReDimNet2-B6:** 12,3 M parametri, 13,05 GMAC. EER VoxCeleb1-O 0,29% con il solo VoxCeleb2, 0,23% con VoxBlink2 + VoxCeleb2 (README del repository, cifre dei proponenti; articolo Interspeech 2026, arXiv 2603.11841).
  - **Perché la variante `vb2+vox2_v0`:** ha l'EER più basso del rilascio su tutti gli insiemi riportati, anche fuori dominio (SITW 0,74%, VOICES 2,55%: VOICES è parlato ripreso a distanza in stanze rumorose). VoxBlink2 aggiunge molte lingue a VoxCeleb2.
  - Le varianti `vox2` (solo VoxCeleb2) e `vb2+vox2+cnc2_v0` restano disponibili (`--voiceprint redimnet2-b6-vox2`).
- **Robustezza allo sforzo vocale:** nei test disponibili la famiglia ReDimNet è la più robusta.
  - Lombard: EER 0,50 / 0,89 contro 1,68 / 2,50 di ECAPA-TDNN.
  - Sussurro: 6,77 contro 10,49.
  - Sono misure su ReDimNet v1; per ReDimNet2 non ci sono ancora dati sullo sforzo vocale.
- **Codec a bitrate molto basso:** degradano tutti i modelli (ECAPA-TDNN da 0,86 a 11,75 con Opus a 1,5 kbps).
- **w2v-BERT 2.0 SV:** ~590 M parametri (580 M del codificatore + 6,2 M di adattatori e testa).
  - Il catalogo riporta Vox1-O 0,12%.
  - Il README del repository riporta 0,14% per il checkpoint pubblicato `model_lmft_0.14.pth` (VoxCeleb2 + VoxBlink2, LoRA + adattatori + MFA, fine-tuning a margine ampio) e 0,18% per la variante con il solo VoxCeleb2. Il codice usa il checkpoint da 0,14%.
- **Nessuna misura** su parlato in auto, autodiretto o italiano per nessun modello.

## Misura su audio italiano reale (2026-11-20)

**Materiale.** 13 clip italiane disponibili offline, ciascuna tagliata in due metà:
- `example-it.wav` di SpeechBrain (Common Voice, 16 kHz, metà di circa 2 s);
- 12 clip it-IT di MInDS-14 (telefono a 8 kHz, metà di 4-9 s).

**Prove.** Stesso parlante = le due metà della stessa clip (13 prove). Parlanti diversi = metà di clip diverse (156 prove). Si assume che clip diverse siano di parlanti diversi; MInDS-14 non lo garantisce.

**Macchina.** CPU condivisa con altri carichi, 4 thread.

| Modello | EER coseno | AUC | d' | coseno medio stesso / diverso | s per embedding |
|---|---|---|---|---|---|
| w2v-BERT 2.0 SV (`model_lmft_0.14`) | 1,0% | 0,997 | 4,46 | 0,77 / 0,26 | 2,09 |
| ECAPA-TDNN (SpeechBrain) | 1,6% | 0,995 | 3,94 | 0,64 / 0,22 | 0,13 |
| ReDimNet2-B6 `vb2+vox2_v0-lm` | 7,7% | 0,989 | 3,70 | 0,75 / 0,28 | 0,94 |
| ReDimNet-B2 v1 `vox2-ft_lm` | 7,4% | 0,988 | 3,52 | 0,65 / 0,19 | 0,12 |
| ReDimNet2-B6 `vox2-lm` | 7,7% | 0,978 | 3,17 | 0,67 / 0,26 | 0,98 |

**Altre misure:**
- **Memoria:** w2v-BERT ha un picco di memoria residente di 2,8 GB (pesi mappati dal file).
- **Dimensioni dell'embedding:** 256 per w2v-BERT, 192 per gli altri.

**Lettura dei risultati:**
- **Campione minimo.** Con 13 prove dello stesso parlante un solo errore vale 7,7 punti di EER: le differenze tra ReDimNet2, ReDimNet-B2 ed ECAPA non sono interpretabili.
- **Coppie difficili comuni.** Le coppie di "parlanti diversi" più simili sono le stesse per tutti i modelli (clip 00, 03, 06, 09, 10). Può trattarsi di voci simili sulla stessa linea telefonica, o dello stesso chiamante.
- **Cosa si può dire:**
  - w2v-BERT separa meglio di tutti anche qui;
  - ReDimNet2-B6 non mostra su questo audio telefonico a 8 kHz il vantaggio dichiarato su VoxCeleb (16 kHz);
  - nessun modello fallisce.
- **Conseguenza.** La scelta predefinita segue le evidenze pubblicate e la decisione del proprietario. Quella per caso la fa l'autovalutazione sul materiale del caso, dove vanno calcolati sia ReDimNet2-B6 sia w2v-BERT (dove la licenza lo consente) ed ECAPA.

## Ricetta di punteggio (WeSpeaker), prima della calibrazione

Implementata in `src/akoe/speaker/score_norm.py` e applicata da `loro_calibration_scores(..., recipe=...)`. Il motore la usa per impostazione predefinita (`build_subject_model(score_recipe="default")`; `None` = solo coseno, come prima).

1. **Sottrazione della media.**
   - Gli embedding vengono normalizzati (norma 1). Si sottrae la media della coorte di altri parlanti e si normalizza di nuovo (WeSpeaker `cal_mean`).
   - I centroidi per condizione sono calcolati nello spazio trasformato.
2. **AS-norm** (normalizzazione simmetrica adattiva).
   - Formula: `0,5 · ((s − μ_e)/σ_e + (s − μ_t)/σ_t)`. μ e σ sono media e deviazione standard dei `top_k` coseni più alti del centroide (lato registrazione) e del segmento (lato prova) con la coorte.
   - Parametri: `top_k = 300` come nella ricetta VoxCeleb di WeSpeaker, ridotto alla dimensione della coorte se questa è più piccola.
3. **QMF** (funzioni di misura della qualità).
   - Regressione logistica (C = 1, classi bilanciate) sul punteggio AS-norm e su tre misure di qualità: logaritmo della durata del segmento, media dei top-k punteggi della coorte per il segmento e per il centroide.
   - Si adatta sulle prove leave-one-recording-out della quota di addestramento.
   - Con meno di 20 prove per classe non si applica, per non adattare quattro coefficienti a pochi dati; si usa il punteggio AS-norm.
4. **Calibrazione** leave-one-recording-out esistente (`calibrate_scores`), sul punteggio così ottenuto.

**Coorte configurabile** (`ScoringRecipe.cohort_sources`). Tre fonti:
- i tratti di altri parlanti etichettati nelle registrazioni di addestramento del caso;
- le clip di altri parlanti date dall'operatore;
- clip di sola coorte (`cohort_files`), per esempio una coorte italiana pubblica calcolata offline.

**Regole della coorte:**
- **Esclusione nei fold.** In ogni fold media, coorte e centroidi escludono il gruppo di registrazioni lasciato fuori. Una clip esterna esce dalla coorte quando viene valutata. Così nessun segmento viene normalizzato contro sé stesso.
- **Coorte piccola.** Con meno di 10 elementi (`min_cohort`) sottrazione della media e AS-norm non si applicano, e il registro lo dice.

**Registrazione.** Ogni parametro, la composizione della coorte, i centroidi trasformati e i coefficienti QMF finiscono nell'artefatto `score-normalization` (SHA-256), referenziato dalla calibrazione. `ProfileScorer` normalizza i nuovi segmenti allo stesso modo. Lo stadio `speaker_scores` registra il tipo di punteggio (`cosine`, `asnorm`, `qmf`).

## Licenze: cosa è stato verificato e cosa no

**Verificato il 2026-11-20:**
- **`PalabraAI/redimnet2`:** licenza MIT (GitHub API e file `LICENSE`), commit `c5bbe0b7` (`main`).
  - I pesi sono allegati della release `v1.0.0` dello stesso repository. Per decisione del proprietario sono coperti dalla licenza del repository.
  - GitHub pubblica il digest SHA-256 di ogni allegato. I tre file B6 `lm` scaricati qui corrispondono, e gli hash sono fissati nel codice:
    - `b6-vb2+vox2_v0-lm.pt` `e0a7d340…e6a2`;
    - `b6-vox2-lm.pt` `bc45f032…23fd`;
    - `b6-vb2+vox2+cnc2_v0-lm.pt` `287365f6…9868`.
  - Il file viene scaricato e verificato prima che `torch.hub` lo apra.
- **`ZXHY-82/w2v-BERT-2.0_SV`** (commit `98ff0e29`): GitHub non riconosce la licenza; il file `License.md` è il testo della **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 (CC-BY-NC-SA-4.0)**.
  - I pesi su Hugging Face (`zl389/w2v-bert-2.0_SV` @ `4e7b92e8`) non dichiarano una licenza propria: per decisione del proprietario vale quella del repository, cioè **uso non commerciale e condivisione allo stesso modo**.
  - Il modello di base `facebook/w2v-bert-2.0` (@ `da985ba0`, di cui si usano solo `config.json` e `preprocessor_config.json`) è MIT.
  - Lo SHA-256 del checkpoint `model_lmft_0.14.pth` è `454398fe…53f9`: è l'oggetto LFS pubblicato dall'hub, verificato sul file scaricato.
- **Confermato dalla prima versione:**
  - ECAPA `speechbrain/spkrec-ecapa-voxceleb` @ `0f99f2d0`: apache-2.0;
  - `IDRnD/ReDimNet` @ `ce039a62`: MIT, SHA-256 dei pesi B2/B6 fissati.

**Non verificato:**
- **Dati di addestramento:** le condizioni d'uso di VoxCeleb2, VoxBlink2 e CN-Celeb, e se vincolino i pesi.
- **Compatibilità con la distribuzione.** La licenza CC-BY-NC-SA di w2v-BERT SV va confrontata con il modello di distribuzione ancora da decidere (`IMPLEMENTATION_PLAN.md` §7). Per questo w2v-BERT non è il predefinito.
- **Brevetti.**

## Requisiti (invariati)

- **Robustezza** al canale (auto, telefono, stanza) e allo sforzo vocale (normale, basso, alzato, sussurrato).
- **Costo su CPU** compatibile con il livello EDGE. ReDimNet2-B6 circa 1 s per un segmento di 4-9 s sulla macchina condivisa; w2v-BERT circa 2 s e 2,8 GB di memoria.
- **Licenza** compatibile con un modello di distribuzione ancora da decidere.
- **Pesi identificabili**, con revisione fissata e SHA-256, per la catena di custodia.

## Prima versione (2026-11-20, issue #9)

La prima versione proponeva ReDimNet-B2 (`b2-vox2-ft_lm`) come principale ed ECAPA come riferimento indipendente. Considerava ReDimNet2 "non valutato" e scartava per ora TitaNet-L (richiede NeMo), WavLM-base-plus-sv (licenza non indicata) ed ECAPA2 (circa 187 GMAC).

Le misure di CPU di allora (mediana su un tratto di 10 s, 4 thread) restano valide per il confronto:
- ECAPA 0,32 s;
- ReDimNet-B2 0,22 s;
- ReDimNet-B6 v1 1,97 s.

**Registrazione per l'impronta (enrollment) da tutte le condizioni (issue #9).** I backend calcolano solo l'embedding di un segmento. Il profilo per condizione, la normalizzazione dei punteggi e la calibrazione restano in `speaker/profile.py` e `speaker/score_norm.py`.


## Aggiornamento del 2026-11-20: w2v-BERT 2.0 SV predefinito

Criterio del proprietario: il migliore disponibile sulle macchine controllate. w2v-BERT 2.0 SV ha l'errore pubblicato più basso (Vox1-O 0,14% per il checkpoint rilasciato) ed è stato il migliore anche sui nostri clip italiani (AUC 0,997, EER 1,0%, contro 7,7% di ReDimNet2-B6; campione piccolo: un errore vale 7,7 punti). Costa circa 2 s per segmento su CPU, accettabile in elaborazione offline. Diventa quindi l'impronta predefinita su tutti i livelli.

ReDimNet2-B6 resta come opzione veloce (`--voiceprint redimnet2-b6`) e come seconda opinione: è la famiglia con la migliore robustezza pubblicata agli sforzi vocali (Lombard, sussurro), che per w2v-BERT non è ancora misurata. L'autovalutazione del caso riporta l'EER sul materiale reale.

Licenza dei pesi di w2v-BERT 2.0 SV: CC-BY-NC-SA-4.0 (dal `License.md` del repository), registrata in `THIRD_PARTY_NOTICES.md`.
