# Decisione: modello per l'impronta vocale (issue #9)

**Data:** 2026-11-20. **Stato:** proposta, da confermare con l'autovalutazione per condizione (M5).

L'impronta vocale in Akoe produce un **punteggio**, mai un'identificazione. Serve a smistare, selezionare e avvisare ("possibile altro parlante"). Non serve per la sintesi della voce (`MASTER_PLAN.md` §3.7).

## Requisiti

- **Robustezza** al canale (auto, telefono, stanza) e allo sforzo vocale (normale, basso, alzato, sussurrato).
- **Costo su CPU** compatibile con il livello EDGE.
- **Licenza** compatibile con un modello di distribuzione ancora da decidere (`IMPLEMENTATION_PLAN.md` §7).
- **Pesi identificabili**, con revisione fissata e SHA-256, per la catena di custodia.

## Candidati

| Modello | Licenza dichiarata | Parametri / GMAC | EER VoxCeleb1-O | CPU misurata qui |
|---|---|---|---|---|
| ECAPA-TDNN, SpeechBrain (`speechbrain/spkrec-ecapa-voxceleb`) | Apache-2.0 (model card) | 20,8 M misurati / 2,67 per C=1024 [1] | 0,80% (model card) | 0,32 s |
| ResNet34-LM, WeSpeaker (`Wespeaker/wespeaker-voxceleb-resnet34-LM`) | CC-BY-4.0 (model card); codice Apache-2.0 | 6,6 M / 4,55 | 0,72-0,87% (model card) | non provato |
| CAM++, WeSpeaker (`Wespeaker/wespeaker-voxceleb-campplus`) | Apache-2.0 (model card) | 7,2 M / 1,15 [1] | 0,71% [1] | non provato |
| ReDimNet-B2, ID R&D (`IDRnD/ReDimNet`, `b2-vox2-ft_lm`) | codice MIT; pesi senza licenza propria | 5,1 M misurati / 0,90 [1] | 0,57% [1] | 0,22 s |
| ReDimNet-B6 (`b6-vox2-ft_lm`) | come B2 | 15,0 M / 20,27 [1] | 0,40% [1] | 1,97 s |

- **CPU misurata qui:** mediana di 5 esecuzioni su un tratto di 10 s, PyTorch 2.15 CPU con 4 thread, macchina condivisa con un altro carico. Le cifre servono a confrontare i modelli, non come tempi assoluti.
- **Fonte [1]:** tabella degli autori di ReDimNet (`EVALUATION.md` del repository). Sono cifre dei proponenti. Per ENFSI non bastano: serve una validazione su dati tipici del caso (`MASTER_PLAN.md` §3.7).
- **ECAPA di SpeechBrain** ha 20,8 M di parametri (canali 1024 e livello finale a 3072), più della variante C=1024 da 14,9 M della tabella [1].

**Altri modelli considerati e scartati per ora:**
- **ReDimNet2** (`PalabraAI/redimnet2`): codice MIT, uscito a luglio 2026, non valutato.
- **TitaNet-L** (`nvidia/speakerverification_en_titanet_large`): CC-BY-4.0, richiede NeMo.
- **WavLM-base-plus-sv** (`microsoft/wavlm-base-plus-sv`): licenza non indicata nella model card.
- **ECAPA2:** circa 187 GMAC stimati, troppo pesante per EDGE.

## Robustezza allo sforzo vocale

- **Sussurrato contro normale:** EER 26,6% per un x-vector contro 6,8% per ReDimNet-B6. **Con il rumore:** EER da 7,8% a 24,5% (Gołębiowska & Syga 2026, come riportato in `MASTER_PLAN.md` §3.5.3; l'articolo non è stato riletto in questa sessione).
- **ECAPA non compare in quei numeri.** Non abbiamo quindi un confronto diretto ECAPA contro ReDimNet sullo sforzo vocale. Il vantaggio documentato è di ReDimNet rispetto agli x-vector.
- **Nessun dato su parlato in auto o autodiretto** per nessuno dei modelli.

## Licenze: cosa è stato verificato e cosa no

**Verificato il 2026-11-20:**
- **Model card su Hugging Face:** campo `license` letto tramite API alle revisioni indicate:
  - `speechbrain/spkrec-ecapa-voxceleb` @ `0f99f2d0`: apache-2.0;
  - `Wespeaker/wespeaker-voxceleb-campplus`: apache-2.0;
  - `Wespeaker/wespeaker-voxceleb-resnet34-LM` e `pyannote/wespeaker-voxceleb-resnet34-LM`: cc-by-4.0;
  - `Wespeaker/wespeaker-voxceleb-resnet221-LM`: apache-2.0;
  - `nvidia/speakerverification_en_titanet_large`: cc-by-4.0;
  - `microsoft/wavlm-base-plus-sv`: nessuna licenza dichiarata.
- **Repository GitHub:** licenza rilevata da GitHub tramite API:
  - `IDRnD/ReDimNet` (commit `ce039a62`): MIT, file `LICENSE` "Copyright (c) 2024 ID R&D, Inc.";
  - `PalabraAI/redimnet2`: MIT;
  - `wenet-e2e/wespeaker`: Apache-2.0;
  - `modelscope/3D-Speaker`: Apache-2.0;
  - `speechbrain/speechbrain`: Apache-2.0.
- **Pesi di ReDimNet:** sono allegati di una release GitHub con tag mutabile `latest` (pubblicata il 2024-06-11), senza digest pubblicati. Gli SHA-256 dei file scaricati oggi sono fissati nel codice:
  - `b2-vox2-ft_lm.pt`: `c9b6bb2f…e06f`;
  - `b6-vox2-ft_lm.pt`: `e0f30343…edea`.
  Un file diverso viene rifiutato.

**Non verificato:**
- se la licenza MIT del repository copra anche i pesi distribuiti come allegati della release (non è detto esplicitamente);
- le condizioni d'uso dei dati di addestramento (VoxCeleb1/2; per le varianti `vb2` e `cnc` di ReDimNet anche VoxBlink2 e CN-Celeb) e se vincolino i pesi;
- il testo completo delle model card oltre al campo licenza;
- eventuali brevetti.

## Raccomandazione

1. **Backend principale proposto: ReDimNet-B2 (`b2-vox2-ft_lm`)**, implementato come `ReDimNetVoiceprint`.
   - **Prestazioni:** EER VoxCeleb dichiarato migliore di ECAPA, circa un terzo dei GMAC, più veloce su CPU nella misura sopra.
   - **Robustezza:** la famiglia ReDimNet è l'unica con un dato pubblicato di robustezza allo sforzo vocale.
   - **Dati di addestramento:** la variante `vox2` usa solo VoxCeleb2 ed evita VoxBlink2 e CN-Celeb.
   - **Integrità:** il codice viene caricato da `torch.hub` a un commit fissato e i pesi sono verificati per SHA-256.
2. **Riferimento indipendente: ECAPA-TDNN di SpeechBrain**, implementato come `EcapaVoiceprint`, a revisione fissata. È la licenza più chiara (Apache-2.0 dichiarata nella model card che contiene i pesi). Nell'autovalutazione si calcolano entrambi.
3. **La scelta per caso la fa l'autovalutazione (M5).** Si confronta l'EER per condizione (auto, telefono, stanza; normale, basso, alzato) sul materiale del caso: stesso soggetto contro altri parlanti presenti.
4. **ReDimNet-B6** è un'opzione su WORKSTATION. È il modello del dato sullo sforzo vocale, ma su CPU è circa 9 volte più lento di B2.
5. **Prima di distribuire fuori dall'uso interno:** chiedere a ID R&D conferma scritta della licenza dei pesi. In mancanza, il ripiego è ECAPA (Apache-2.0).
6. **WeSpeaker CAM++** (Apache-2.0 nella model card) resta il terzo candidato. Richiede il pacchetto `wespeaker` o `onnxruntime`, non presenti nell'ambiente; va valutato solo se ReDimNet non regge nell'autovalutazione.

**Registrazione per l'impronta (enrollment) da tutte le condizioni (issue #9):** i backend calcolano solo l'embedding di un segmento (192 dimensioni). Il profilo per condizione, il punteggio coseno e la calibrazione restano allo stadio `speaker/voiceprint.py` (M1).
