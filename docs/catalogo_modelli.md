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
| ReDimNet2-B6 (PalabraAI, 2026) | Vox1-O EER 0,23–0,29%, 12,3M parametri | MIT | **Predefinito su tutti i livelli**; la famiglia ReDimNet è la più robusta a Lombard e sussurro nei test disponibili |
| w2v-BERT 2.0 + MFA (2025) | Vox1-O EER 0,12%, ~600M | repository (licenza del repository) | Massima accuratezza su WORKSTATION/CLOUD; su EDGE solo offline |
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
| UniPASE (2026) | dWER più basso anche con riverbero (8,16 contro 14,16 di AnyEnhance): meno allucinazioni | MIT | WORKSTATION/CLOUD; EDGE offline se i tempi lo consentono |
| MossFormer2_SE_48K | VoiceBank PESQ 3,15, SI-SDR 19,36 | Apache-2.0 | WORKSTATION |
| GTCRN | 48K parametri, RTF 0,07 su CPU | MIT | EDGE, rapido |
| DeepFilterNet 3 | robusto e leggero | MIT/Apache-2.0 | EDGE, confronto |

Esclusi i generativi con molte allucinazioni documentate (StoRM, LLaSE-G1).

## Rilevamento della voce

| Modello | Licenza | Uso |
|---|---|---|
| Silero VAD 6.2.3 | MIT (PyPI) | Opzione neurale su tutti i livelli |
| FireRedVAD | Apache-2.0 | Confronto |
| VAD deterministico di Akoe | SANSEVIERIA 1.1 | Opzione riproducibile, predefinita in modalità forense |

## Adattamento al soggetto

Oggi su Whisper (LoRA/completo). Da estendere a Qwen3-ASR (fine-tuning nel repository), Parakeet e Nemotron (ricette NeMo), così il modello del soggetto si costruisce sulle famiglie più forti.
