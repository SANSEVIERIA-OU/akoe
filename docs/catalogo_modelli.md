# Catalogo dei modelli per livello (novembre 2026)

Criterio del proprietario (2026-11-20): **il migliore disponibile al mondo**, con il solo vincolo di girare su macchine controllate. Fonti: ricerche del 2026-11-20 su schede dei modelli, report tecnici e classifiche (dettaglio e limiti in `MASTER_PLAN.md` §3.11). I numeri italiani vengono da snapshot diversi della classifica Open ASR (media su più dataset) e non sono confrontabili al decimale.

Livelli: **EDGE** = portatile senza GPU (14 GB RAM); **WORKSTATION** = una GPU NVIDIA; **CLOUD** = VM con GPU dell'utente.

## Trascrizione (più famiglie insieme: alternative e controllo incrociato delle allucinazioni)

| Modello | Italiano | Dimensione, licenza | EDGE | WS | CLOUD | Note |
|---|---|---|---|---|---|---|
| NVIDIA Nemotron 3.5 Transcribe | media 3,61; FLEURS-it 1,39 | 34B (4,3B attivi), OpenMDW-1.1 | — | — | ✔ principale | ~67 GB in BF16, una H100; timestamp per parola; ricetta LoRA in NeMo |
| Cohere Transcribe 03-2026 | media 3,44 (marzo) | 2B, Apache-2.0 | ✔ | ✔ | ✔ | Il miglior italiano aperto che gira ovunque; confidenza/N-best da costruire sul decoder |
| Qwen3-ASR-1.7B | FLEURS-it 2,41 | 1,7B, Apache-2.0 | ✔ | ✔ | ✔ | Il più robusto al rumore nel test URGENT (inglese); allineatore forzato con italiano; fine-tuning nel repository |
| NVIDIA Parakeet-TDT 0.6B v3 | FLEURS-it 3,00; media 4,81 | 0,6B, CC-BY-4.0 | ✔ | ✔ | ✔ | Confidenza entropica nativa (NeMo); veloce |
| Meta omniASR-LLM-7B | media 3,75 | 7,8B, Apache-2.0 | — | ✔ | ✔ | ~17 GB di VRAM |
| Mistral Voxtral Small 24B | media 3,75 | 24B, Apache-2.0 | — | — | ✔ | |
| OpenAI Whisper large-v3 | media 4,45 | 1,55B, MIT | ✔ | ✔ | ✔ | Resta per l'adattamento al soggetto (strumenti LoRA già pronti) e come famiglia di confronto |

## Impronta vocale

| Modello | Prestazioni | Licenza | Uso |
|---|---|---|---|
| w2v-BERT 2.0 + MFA (2025) | Vox1-O EER 0,14% (checkpoint rilasciato); sui nostri clip italiani AUC 0,997, EER 1,0% | CC-BY-NC-SA-4.0 | **Predefinito su tutti i livelli** (~2 s per segmento su CPU, offline) |
| ReDimNet2-B6 (PalabraAI, 2026) | Vox1-O EER 0,23–0,29%, 12,3M parametri; sui nostri clip EER 7,7% (campione piccolo) | MIT | Opzione veloce e seconda opinione; la famiglia ReDimNet è la più robusta a Lombard e sussurro nei test pubblicati |
| ReDimNet-B6 v1 / ECAPA-TDNN | riferimento | MIT / Apache-2.0 | Confronto e ripiego |

Punteggi: coseno con sottrazione della media, **AS-norm** e **QMF** (ricetta WeSpeaker), poi calibrazione leave-one-recording-out.

## Allineamento forzato e frasi candidate

| Modello | Italiano | Licenza | Uso |
|---|---|---|---|
| omniASR CTC 1B v2 / 3B v2 (Meta) | FLEURS-it 3,9% / 1,3% (campione minimo) | Apache-2.0 | Punteggio "X o Y?": 1B su EDGE, 3B su WS/CLOUD |
| jonatasgrosman/wav2vec2-xls-r-1b-italian | CV8 WER 9,04, CER 2,20 | Apache-2.0 | Allineamento a caratteri dei testi del soggetto (sostituisce il large-xlsr-53) |
| Qwen3-ForcedAligner | italiano tra le 11 lingue | Apache-2.0 | Timestamp delle trascrizioni Qwen3-ASR |

## Pulizia d'ascolto neurale (mai in ingresso al trascrittore senza verifica)

| Modello | Evidenza | Licenza | Uso |
|---|---|---|---|
| MossFormer2_SE_48K | Sui nostri clip italiani in rumore d'auto: SI-SDR +12,5 dB, ESTOI +0,148, WER di Whisper da 0,327 a 0,286; ~0,4× tempo reale su CPU | Apache-2.0 | **Predefinito** su tutti i livelli (su EDGE accanto a GTCRN) |
| UniPASE (2026) | dWER più basso nella letteratura, ma sui nostri clip risintetizza (SI-SDR −9,6 dB) e peggiora il WER (0,406) | MIT | Solo su richiesta |
| GTCRN | 48K parametri; sui nostri clip SI-SDR +10,2 dB, 0,05× tempo reale | MIT | EDGE, versione d'ascolto rapida |
| DeepFilterNet 3 | robusto e leggero | MIT/Apache-2.0 | EDGE, confronto |

Esclusi i generativi con molte allucinazioni documentate (StoRM, LLaSE-G1).

La scelta predefinita è stata cambiata da UniPASE a MossFormer2 dopo la misura su 7 clip italiani reali in rumore d'auto (0/5/10 dB e riverbero d'abitacolo).

## Rilevamento della voce

| Modello | Licenza | Uso |
|---|---|---|
| FireRedVAD | Apache-2.0 | Opzione neurale (F1 0,954 a 0 dB nei nostri test, con 6% di falsi allarmi) |
| Silero VAD 6.2.3 | MIT (PyPI) | Opzione neurale (F1 0,903 a 0 dB) |
| VAD deterministico di Akoe | SANSEVIERIA 1.1 | Predefinito; **l'unico ammesso in modalità forense** (F1 0,913 a 0 dB) |

## Adattamento al soggetto

Oggi su Whisper (LoRA/completo). Da estendere a Qwen3-ASR (fine-tuning nel repository), Parakeet e Nemotron (ricette NeMo), così il modello del soggetto si costruisce sulle famiglie più forti.
