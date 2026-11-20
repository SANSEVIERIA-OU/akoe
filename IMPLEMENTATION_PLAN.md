# Implementation Plan — WHISPER

Documento operativo. Chiunque (umano o Claude in una sessione futura) deve poter aprire questo file e capire dove siamo, dove andiamo e perché. Contesto, evidenze e motivazioni sono in `MASTER_PLAN.md`.

**Revisione 2 — 2026-11-20.** Sostituisce il piano del 2026-05-25, che era organizzato per tier attorno a una pipeline generalista con estrazione del parlante.

Le ricerche del 2026-11-20 sull'analisi del timbro, sul rapporto tra pulizia e trascrittore e sulle interpretazioni multiple sono integrate (`MASTER_PLAN.md`, sezioni 3.4-3.6). I punti ancora aperti sono segnati come **Da verificare**.

**Repository e roadmap:** `github.com/SANSEVIERIA-OU/whisper` (privato). Le fasi M0-M8 sono milestone e i lavori sono issue; questo documento resta la descrizione di riferimento.

**Modello del soggetto multi-contesto** (idea guida dell'utente): impronta vocale e adattatore del trascrittore si addestrano su tutte le occasioni in cui il soggetto parla (telefono, stanze, auto), con simulazione di canali, codec e sforzi vocali. L'obiettivo è riconoscerlo e interpretarne le parole indipendentemente dal contesto; l'autovalutazione misura per condizione dove ci riesce.

**Microspia:** registra in locale e trasmette, con codec variabili. Ogni evento può avere due reperti (locale e trasmesso); la registrazione locale è l'ingresso predefinito.

**Principio guida emerso dalle ricerche.** La conoscenza del timbro del soggetto e dell'acustica dell'auto va **dentro il trascrittore automatico** (adattamento). La pulizia personalizzata serve all'**ascolto** del trascrittore umano. Al trascrittore automatico va l'audio originale, o il pulito rimescolato con l'originale, scelto caso per caso dall'autovalutazione.

---

## 1. Obiettivo e scope

**Prodotto sperimentale** per la trascrizione assistita di intercettazioni ambientali. Scenario primario: **soggetto da solo in auto**, anche in soliloquio, un solo microfono nascosto.

**Cosa fa il prodotto:**
1. costruisce un **profilo del soggetto** da sue registrazioni reali: impronta vocale, profilo timbrico-spettrale, dati di adattamento, lessico;
2. costruisce un **profilo del veicolo**: risposta all'impulso e rumore, misurati o stimati;
3. pulisce l'audio con uno strato **riproducibile** e, come ipotesi, uno strato **neurale**;
4. **adatta il trascrittore automatico** al soggetto e al canale;
5. produce **interpretazioni multiple** con punteggi, confronta frasi candidate ("ha detto X o Y?") e sa dire "incomprensibile";
6. accompagna ogni caso con un'**autovalutazione**;
7. offre un'interfaccia di **revisione cieca** (app Tauri) e un'esportazione per la perizia.

**Fuori scope per ora:**
- scenari con più parlanti: l'estrazione del parlante (`extract.py`) è accantonata;
- sintesi o clonazione della voce del soggetto: esclusa (`MASTER_PLAN.md` 4.2).

**Metodo:** si costruisce il prodotto. La misura delle prestazioni è una funzione del prodotto (autovalutazione), non un esperimento separato.

---

## 2. Architettura

```
┌──────────────────────────┐        API (HTTP locale / rete cifrata)        ┌──────────────────────────────┐
│  App desktop Tauri       │ ─────────────────────────────────────────────▶ │  Motore Python forensicaudio │
│  (frontend web, Rust)    │ ◀───────────────────────────────────────────── │  (stadi, modelli, log)       │
└──────────────────────────┘                                                └──────────────────────────────┘
        sempre sul computer dell'operatore                     EDGE: stesso computer (sidecar)
                                                               WORKSTATION / CLOUD: macchina con GPU controllata
```

- **Una sola interfaccia per tutti i livelli.** Cambia solo dove gira il motore.
- **"Si addestra dove c'è potenza, si usa ovunque".** Gli adattatori prodotti su workstation o cloud sono file piccoli, versionati e con SHA-256, utilizzabili anche sul portatile.
- **Assi `Mode` (FORENSIC / INVESTIGATIVE / BOTH) e `Tier` (EDGE / WORKSTATION / CLOUD)** restano ortogonali.
- **I componenti che devono girare bene su CPU** si possono portare in Rust con ONNX Runtime, con esportazione verificata numericamente, sul modello di `/workspace/kalliope/tools/pocket_export/` (M8).

---

## 3. Moduli del motore

| Modulo | File previsto | Tipo | Stato | Fase |
|---|---|---|---|---|
| Caso, copia di lavoro, catena di custodia | `pipeline.py`, `utils.py`, `models.py` | DETERMINISTIC | esiste | M0 (estensione) |
| Preprocessing (16 kHz mono, normalizzazione) | `stages/preprocess.py` | DETERMINISTIC | esiste | — |
| Rilevamento della voce | `stages/vad.py` | DETERMINISTIC | esiste (a energia) | M6 (miglioramento) |
| Riduzione del rumore stazionaria | `stages/denoise.py:run_deterministic` | DETERMINISTIC | esiste | M6 (estensione) |
| Dati del soggetto (import, allineamento, divisione) | `speaker/dataset.py` | DETERMINISTIC | da fare | M1 |
| Impronta vocale | `speaker/voiceprint.py` (da `stages/embedding.py`) | NEURAL (pesi fissi) | stub | M1 |
| Profilo timbrico-spettrale | `speaker/timbre.py` | DETERMINISTIC | da fare | M1 |
| Trascrizione con alternative | `transcribe/asr.py` (da `stages/asr.py`) | NEURAL | stub | M2 |
| Punteggio delle frasi candidate | `transcribe/candidates.py` | NEURAL | da fare | M2 |
| Profilo del veicolo | `vehicle/profile.py` | DETERMINISTIC | da fare | M3 |
| Simulazione del canale | `vehicle/simulate.py` | DETERMINISTIC | da fare | M3 |
| Adattamento del trascrittore | `adapt/train.py`, `adapt/artifact.py` | NEURAL | da fare | M4 |
| Lessico del caso | `adapt/lexicon.py` | NEURAL (influenza il decoding) | da fare | M4 |
| Autovalutazione | `assess/selfcheck.py` | DETERMINISTIC (misura) | da fare | M5 |
| Filtri per bande e armoniche del motore | `stages/bands.py`, `stages/harmonics.py` | DETERMINISTIC | da fare | M6 |
| Cancellazione con riferimento (radio/musica) | `stages/reference_cancel.py` | DETERMINISTIC | da fare | M6 |
| Pulizia personalizzata | `stages/denoise.py:run_neural` | NEURAL | stub | M6 |
| API del motore | `api/server.py` | — | da fare | M7 |
| Esportazione per la perizia | `export/perizia.py` | DETERMINISTIC | da fare | M7 |
| App desktop | `app/` (progetto Tauri) | — | da fare | M7 |
| Estrazione del parlante (TSE) | `stages/extract.py` | NEURAL | **accantonato** | — |

---

## 4. Stato attuale del codice

Verificato il 2026-05-25 sulla macchina di test (Ryzen 5 PRO 6650U, 14 GB RAM, senza GPU); il codice è invariato al 2026-11-20.

**Funziona end-to-end in `mode=FORENSIC`:**
- preprocessing (`stages/preprocess.py`);
- riduzione del rumore stazionaria (`stages/denoise.py:run_deterministic`, noisereduce);
- rilevamento della voce a energia su frame di 30 ms (`stages/vad.py`);
- orchestratore con SHA-256 di input e output per stadio (`pipeline.py`);
- CLI `forensicaudio process` (`--mode`, `--evidence`, `--speaker-samples`, `--output`, `--case-id`);
- 2 test pytest verdi.

**Non implementati:** `denoise.run_neural`, `embedding.run`, `extract.run` e `asr.run` sollevano `NotImplementedError`, registrato nel log come "skipped".

---

## 5. Fasi

Ordine scelto: prima il cuore del prodotto (profilo del soggetto → alternative → canale → adattamento → autovalutazione), poi la pulizia specifica per l'auto, poi interfaccia e distribuzione. L'interfaccia (M7) può partire in parallelo appena esiste l'API minima di M2.

### M0 — Fondamenta

- **Tier:** enum `Tier` in `config.py`, `CaseConfig.tier`, flag `--tier`; nello `StageLog` i campi `tier`, `device`, `model_sha256`.
- **Modelli dati** in `models.py`:
  - `Alternative` e `WordHypothesis` (testo, tempi, punteggio, origine);
  - `SegmentHypotheses`;
  - `SpeakerProfile`, `VehicleProfile`, `AdapterArtifact`, `SelfAssessment`;
  - provenienza per parola nell'esportazione.
- **Struttura del caso su disco:** `cases/<id>/{evidence,working,profiles,adapters,outputs,logs}`; l'evidenza originale è in sola lettura, con hash all'import.
- **Pipeline:** profilo di esecuzione `vehicle-single-speaker` che esclude `extract.py`; lo stub resta nel repository.
- **Catena di custodia** estesa a profili, adattatori e lessico.
- **Fatto quando:** test esistenti verdi più nuovi test sui modelli dati e sulla struttura del caso.

### M1 — Profilo del soggetto

- **Dati del soggetto** (`speaker/dataset.py`):
  - import di registrazioni del soggetto e, se ci sono, delle trascrizioni esistenti (testo semplice, SRT, JSON);
  - **condizione** di ogni registrazione: canale (auto, telefono, stanza, altro), sforzo vocale, sorgente (locale o trasmessa), codec rilevato; registrazioni locale e trasmessa dello stesso evento collegate da un identificativo di coppia;
  - allineamento delle trascrizioni all'audio per ritagliare segmenti testo+audio, segnalando quelli male allineati;
  - divisione addestramento / autovalutazione **per registrazione**, non per segmento, per evitare che la stessa sessione finisca da entrambe le parti; quota predefinita 20%;
  - controllo di qualità delle trascrizioni (dialetto italianizzato, parti illeggibili), con segnalazione all'operatore.
- **Impronta vocale** (`speaker/voiceprint.py`):
  - **Modello:** da scegliere tra ECAPA-TDNN (SpeechBrain), ResNet o CAM++ (WeSpeaker), dopo verifica di licenza e prestazioni su CPU.
  - **Punteggi:** un punteggio per segmento contro il profilo; calibrazione sul materiale del caso (stesso parlante contro altri parlanti presenti nel materiale).
  - **Selezione assistita:** proposta dei tratti del soggetto in altre registrazioni, sempre con conferma dell'operatore.
  - **Avvisi:** "possibile altro parlante" quando il punteggio scende.
  - **Mai** "identificazione" nell'interfaccia o nell'esportazione.
- **Profilo timbrico-spettrale** (`speaker/timbre.py`):
  - **Misure:** spettro medio di lungo periodo, distribuzione di F0 (stabile dopo circa 10 s di parlato), intervalli delle formanti con misura automatica e parametri registrati, solo su tratti con SNR sufficiente.
  - **Profili per condizione** (canale: telefono, stanza, auto; sforzo: normale, basso, alzato), etichettati dall'operatore o stimati. Ogni punteggio dichiara la condizione di riferimento (`MASTER_PLAN.md` 3.5.3).
  - **Usi del profilo:**
    - parametri dei filtri per bande (M6);
    - modelli della voce per la pulizia d'ascolto: dizionari NMF del soggetto più dizionario del rumore dell'auto come metodo ispezionabile, codebook di inviluppi da circa 40 s come post-filtro (`MASTER_PLAN.md` 3.5.1);
    - verifica che i dati di adattamento siano coerenti con la condizione del caso.
  - **Ciclo di verifica:** i tratti di registrazioni precedenti nella stessa auto marcati "certi" dal trascrittore entrano nel profilo e nei dati di adattamento, con versione. I tratti incerti mai.
  - **Da verificare:** durata minima per uno spettro medio stabile (non trovata in letteratura); gestione del parlato autodiretto (nessun dato).
- **Fatto quando:** da una cartella di registrazioni del soggetto si ottiene un `SpeakerProfile` con hash, una divisione registrata e un report di qualità.

### M2 — Trascrizione con interpretazioni multiple (modello generico)

- **Modelli** (proposta, `MASTER_PLAN.md` 3.4.3 e 3.6):
  - **Parakeet-TDT 0.6B v3** (NeMo, CC-BY-4.0) come principale: WER italiano basso, confidenza basata sull'entropia (`confidence_cfg`, Tsallis α = 0,33), dimensioni adatte alla CPU;
  - **Whisper large-v3 / turbo** (MIT) come seconda famiglia: errori diversi, strumenti LoRA diffusi. Sul portatile: Whisper small/turbo quantizzato;
  - **un modello CTC italiano** per l'allineamento forzato e il punteggio delle candidate.
  - **Da verificare:** se Parakeet-TDT v3 ha una testa CTC usabile da NeMo Forced Aligner (supporta solo CTC o ibridi in modalità CTC); quale modello CTC italiano usare e con quale licenza.
- **Alternative:**
  - **fonti interpretabili:** modello generico contro adattato (da M4); Parakeet contro Whisper; con lessico contro senza (da M4); audio originale contro rimescolato (da M6);
  - **per ampliare la scelta:** campionamento con scelta MBR e le N migliori ipotesi. Il beam di Whisper ha poca diversità (HyPoradise). Con l'MBR conviene tenere tutti i candidati, non solo il vincitore, perché sul parlato conversazionale può peggiorare (Jinnai 2025);
  - **allineamento delle ipotesi** parola per parola (stile ROVER), mostrando per ogni parola le varianti con la loro origine;
  - **confidenza** mostrata per fasce, calibrata nell'autovalutazione (M5).
- **Frasi candidate** (`transcribe/candidates.py`):
  - dato un segmento e da 2 a circa 5 frasi scritte dall'operatore, un punteggio di compatibilità per ciascuna, più "nessuna compatibile";
  - **metodo:** punteggio di allineamento forzato con il modello CTC, più la verosimiglianza della frase forzata nel decoder di Whisper; i due punteggi vanno mostrati separati;
  - **soglie** tarate nell'autovalutazione (frase giusta contro frasi alterate), perché la correlazione tra punteggio ed errore cala con molti candidati (Novosad 2026) e su audio scadente l'allineamento può "sembrare funzionare" (Loakes 2022);
  - **nota tecnica:** `torchaudio.functional.forced_align` è indicato come deprecato dalla 2.8 e in rimozione nella 2.9. Fissare la versione o usare NeMo Forced Aligner o CTC-segmentation.
- **"Incomprensibile":** etichetta esplicita sotto una soglia tarata.
- **Allucinazioni:**
  - testo prodotto su tratti che il rilevatore di voce considera silenzio o rumore (su non-voce Whisper large-v3 produce testo nel 40% dei casi);
  - lista di frasi tipiche di allucinazione (approccio "Bag of Hallucinations", Barański et al. 2025);
  - forte disaccordo tra famiglie di modelli.
  - Tutto segnalato, mai rimosso in silenzio.
- **Fatto quando:** `transcript.json` con alternative, confidenza e provenienza `NEURAL` per ogni segmento, e un endpoint per il punteggio delle candidate.

### M3 — Profilo del veicolo e simulazione del canale

- **Runbook di misura** (`docs/runbook_misura_veicolo.md`):
  - riproduzione di un segnale di misura (sweep sinusoidale) da un altoparlante all'altezza della testa del guidatore, registrato dalla microspia o da un microfono dello stesso modello nella stessa posizione;
  - registrazioni di rumore per condizione: motore spento, minimo, città, extraurbano, autostrada, finestrini chiusi e aperti, ventilazione accesa e spenta;
  - metadati: veicolo, posizione del microfono, data, operatore.
- **Profilo** (`vehicle/profile.py`): import di risposte all'impulso e rumori, con hash; in assenza di misure, profilo del rumore stimato da tutti i tratti senza voce della registrazione, con traccia delle armoniche del motore.
- **Simulazione** (`vehicle/simulate.py`):
  - parlato pulito del soggetto convoluto con la risposta all'impulso, **con compensazione del ritardo** (una convoluzione disallineata aumenta gli errori; Tang et al.);
  - più **rumore registrato di quell'auto** a rapporti segnale/rumore campionati da una distribuzione stimata sul caso. Il rumore reale conta quanto la risposta all'impulso: con rumore corrispondente, risposte artificiali e reali rendono quasi uguale (Szöke et al. 2019);
  - **codec variabili:** la microspia trasmette con codec che possono cambiare. Si simula un insieme di codec: G.711 μ-law/A-law, MP3, Opus e Vorbis tramite libsndfile; AMR e GSM solo se c'è un encoder ffmpeg, altrimenti saltati con avviso nel log. Si sceglie il codec a caso per esempio di addestramento, perché la simulazione di codec può peggiorare le condizioni che non lo prevedono (Balam et al. 2020);
  - **coppie locale/trasmesso:** allineamento temporale delle due registrazioni dello stesso evento; stima del canale di trasmissione (banda, codec, perdite); le coppie reali entrano nell'autovalutazione;
  - **rilevamento del codec** dai metadati dei file, quando possibile;
  - **risposta del microfono:** nessuna evidenza, non prioritaria.
- **Fatto quando:** da un `VehicleProfile` e da segmenti puliti si genera un insieme simulato riproducibile (seme registrato, hash degli output).

### M4 — Adattamento del trascrittore

- **Dati (multi-contesto):** tutti i segmenti di addestramento verificati del soggetto (M1), in tutte le condizioni. Su di essi si applicano simulazioni di canali diversi (auto misurata, stanze, telefono), codec variabili e rumori diversi (M3). Una quota resta pulita, per non degradare il modello sul parlato normale.
- **Struttura:** un adattatore del soggetto multi-contesto; facoltativamente una specializzazione leggera per condizione (per esempio quell'auto), solo se l'autovalutazione per condizione mostra un guadagno.
- **Addestramento** (`adapt/train.py`):
  - su WORKSTATION o CLOUD;
  - **LoRA come impostazione predefinita**, con rango non minimo: con rango 8 su un singolo parlante il risultato è peggiore del 15-39% relativo rispetto al fine-tuning completo (Huber & Waibel 2026a), mentre in altri casi LoRA e completo sono quasi pari (Fan et al. 2024: 9,6% contro 9,3%);
  - **fine-tuning completo** come opzione su workstation o cloud, scelto se l'autovalutazione mostra un guadagno;
  - **memoria:** esecuzioni documentate di LoRA su Whisper large tra 12 e 20 GB di GPU; non esistono cifre ufficiali. Va misurata e registrata nel log;
  - **EDGE:** solo modelli piccoli (tiny/small), con tempi lunghi dichiarati all'operatore; nessun tempo misurato senza GPU in letteratura;
  - **adattamento al momento della trascrizione** (test-time adaptation): opzione futura, costosa e con risultati non sempre positivi (`MASTER_PLAN.md` 3.4.3).
- **Artefatto** (`adapt/artifact.py`): modello di base, versione, hash dei dati di addestramento, iperparametri, seme, SHA-256 dei pesi; caricabile su qualsiasi livello.
- **Lessico del caso** (`adapt/lexicon.py`):
  - costruito da trascrizioni precedenti e confermato dall'operatore;
  - usato con biasing contestuale con filtro contro gli inserimenti (per esempio il biasing a trie con filtro sillabico di Liu et al. 2025, che senza filtro porta gli errori sulle altre parole dall'1,9% al 21,2%); implementazione da scegliere in base al modello;
  - obbligatori il test di inserimento (M5) e la doppia ipotesi "con / senza lessico" quando divergono.
- **Fatto quando:** un adattatore addestrato su workstation trascrive sul portatile, e l'autovalutazione mostra il confronto tra modello generico e adattato.

### M5 — Autovalutazione per caso

- **Input:** quota di autovalutazione (M1), simulata con il profilo del veicolo (M3) al rapporto segnale/rumore stimato del caso, più le coppie reali locale/trasmesso quando ci sono.
- **Per condizione:** ogni misura è riportata separatamente per condizione (canale, sorgente, codec, sforzo), così si vede dove il modello del soggetto regge e dove no.
- **Misure:**
  - WER e CER del modello generico e di quello adattato;
  - copertura delle alternative (WER oracolo sulle N migliori);
  - calibrazione della confidenza (ECE, diagramma di affidabilità);
  - tasso di "incomprensibile" corretto;
  - tasso di inserimento del lessico su parole volutamente assenti;
  - EER e calibrazione dell'impronta vocale;
  - affidabilità del punteggio delle candidate (frase giusta contro frasi alterate).
- **Output:** `self_assessment.json` e una sintesi leggibile allegata alla trascrizione ("su materiale simile, in queste condizioni, circa X parole su 100 sono sbagliate"), con il limite dichiarato (`MASTER_PLAN.md` 4.7).
- **Fatto quando:** ogni trascrizione esportata include la propria autovalutazione, oppure dichiara perché non è stato possibile calcolarla (per esempio, nessun materiale trascritto).

### M6 — Pulizia specifica per l'auto

- **Strato riproducibile:**
  - filtraggio per bande motivato dall'SII e dal profilo spettrale del soggetto (`stages/bands.py`);
  - riduzione del rumore con profilo stimato sull'intera registrazione (estensione di `run_deterministic`);
  - tracciamento e attenuazione delle armoniche del motore (`stages/harmonics.py`);
  - cancellazione con riferimento (`stages/reference_cancel.py`): identificazione del brano o della trasmissione per impronta acustica (strumenti open-source da valutare), allineamento, filtro adattivo. Il riverbero dell'abitacolo impedisce la sottrazione semplice (Alexander et al. 2012); attenzione a non eliminare la voce se trapela nel riferimento.
  - Tutti i parametri nel log, in linea con SWGDE 20-A-001-2.0.
- **Strato neurale** (`run_neural`): pulizia informata dal parlante, **per l'ascolto del trascrittore umano**, etichettata come neurale.
  - Prima scelta ispezionabile: dizionari NMF della voce del soggetto più dizionario del rumore misurato dell'auto.
  - Opzione neurale: modello di pulizia rifinito sul parlato del soggetto, con rumore dell'auto simulato.
  - Evidenze: guadagni moderati e reali per gli ascoltatori (Goehring et al. 2017; Leer et al. 2026); nessuno studio nelle condizioni esatte del caso (`MASTER_PLAN.md` 3.5.1).
- **Cosa va al trascrittore automatico:**
  - per impostazione predefinita l'audio originale, al più dopo cancellazione della radio e filtri lievi;
  - in alternativa il pulito **rimescolato** con una quota dell'originale (per esempio 10-25%), che negli studi recupera il danno della pulizia e talvolta migliora di poco (`MASTER_PLAN.md` 3.4.1);
  - variante e quota scelte per caso dall'autovalutazione. Mai il pulito da solo senza verifica: davanti a Whisper peggiora nella maggior parte degli studi.
- **Rilevamento della voce:** sostituire quello a energia con uno più robusto al rumore d'auto, mantenendo l'opzione deterministica.
- **Fatto quando:** l'autovalutazione riporta l'effetto di ogni strato di pulizia sul WER, e l'operatore può ascoltare ogni versione etichettata.

### M7 — API del motore e app Tauri

- **API** (`api/server.py`): casi, import, avvio di job con avanzamento, profili, trascrizioni con alternative, punteggio delle candidate, tasselli di spettrogramma calcolati dal motore con parametri registrati, streaming audio delle versioni etichettate, autovalutazione, esportazione.
- **App** (`app/`, Tauri):
  - vista caso e stato dei job;
  - **editor di trascrizione cieca**: forma d'onda e spettrogramma, ripetizione di un tratto, rallentamento, alternanza originale/pulito; le ipotesi della macchina si sbloccano solo dopo la prima trascrizione dell'operatore per quel tratto;
  - pannello delle alternative per parola e per segmento;
  - pannello "ha detto X o Y?";
  - pannello dell'autovalutazione;
  - esportazione per la perizia.
  - Italiano per primo, accessibile da tastiera.
- **Esportazione** (`export/perizia.py`): testo del trascrittore con provenienza per parola, alternative scartate, riferimenti al log, autovalutazione, versioni di modelli e adattatori.
- **Fatto quando:** un operatore apre un caso, trascrive alla cieca un tratto, confronta le ipotesi, interroga "X o Y?" ed esporta, sul portatile con motore locale.

### M8 — Livelli hardware e distribuzione

- **EDGE:**
  - motore come sidecar dell'app (impacchettamento del Python da valutare);
  - gestore dei modelli con download separato e verifica degli hash;
  - componenti CPU-critici (impronta vocale, eventualmente trascrittore piccolo) esportabili in ONNX ed eseguibili da Rust, con verifica numerica contro il riferimento come in Kalliope.
- **WORKSTATION:** rilevamento di CUDA con errore esplicito se manca; device e picco di memoria nel log.
- **CLOUD:** motore remoto su VM controllata, connessione cifrata, vincoli della sezione 8.
- **Requisiti noti:** LoRA su Whisper large documentato tra 12 e 20 GB di GPU; senza GPU, adattamento realistico solo per modelli piccoli; Parakeet-TDT 0.6B adatto all'inferenza su CPU. **Da misurare** sul portatile di riferimento e su una workstation tipo, con i risultati registrati in questo documento.
- **Fatto quando:** lo stesso caso si apre dal portatile sia con motore locale sia con motore remoto, con log che riportano livello e device.

---

## 6. Invarianti (validi su tutti i livelli)

1. **Separazione probatoria:** audio deterministico e neurale mai uniti in un unico artefatto; `audio_provenance` sempre presente.
2. **Catena di custodia:** SHA-256 di input e output per ogni stadio, e anche per profili, adattatori, lessico e autovalutazione.
3. **Riproducibilità:** gli stadi deterministici danno output identico su input identico; quelli neurali registrano modello, pesi (hash), seme e parametri.
4. **Etichettatura AI:** ogni output neurale è marcato come ipotesi.
5. **Niente crash sugli stadi non implementati:** `NotImplementedError` viene registrato come "skipped".
6. **Niente SaaS di terzi:** l'audio passa solo per macchine controllate dall'utente.
7. **Nessuna sintesi della voce del soggetto.**
8. **Revisione cieca come impostazione predefinita** nell'interfaccia.
9. **Impronta vocale come punteggio**, mai come identificazione.
10. **Ogni parola esportata** ha provenienza e alternative.

---

## 7. Decisioni rimandate

- **Modello di distribuzione** (open source, uso interno, commerciale): determina le licenze accettabili.
- **Conferma dei trascrittori proposti** (Parakeet-TDT v3, Whisper large-v3, modello CTC italiano) dopo le prime autovalutazioni; disponibilità di una testa CTC per Parakeet.
- **Modello dell'impronta vocale** e relativa licenza; robustezza allo sforzo vocale (i modelli più recenti reggono molto meglio degli x-vector, `MASTER_PLAN.md` 3.5.3).
- **Pulizia d'ascolto:** NMF del soggetto o modello neurale rifinito, dopo il confronto sul banco di prova.
- **Effetto sui trascrittori umani** della presentazione delle alternative: nessuno studio controllato, da osservare nel prodotto.
- **Strumentazione di misura** del veicolo e formato della microspia (registrazione locale o trasmissione compressa).
- **Inquadramento legale:** AI Act, dati biometrici, trattamento dei dati in ambito di polizia.
- **Scenari con più parlanti:** ritorno di diarizzazione ed estrazione del parlante.
- **Materiale reale** per il banco di prova.

---

## 8. Vincoli per il livello CLOUD (non negoziabili)

- La VM è **sotto controllo dell'utente**; nessun servizio SaaS che elabori l'audio sui propri server (OpenAI API, AssemblyAI, servizi di isolamento vocale e simili).
- Disco cifrato, accesso SSH con chiave, connessione cifrata tra app e motore.
- A fine sessione: scaricamento cifrato degli artefatti, cancellazione sicura del disco e distruzione della VM, secondo un runbook dedicato.
- `processing_log.json` registra provider, regione, ID istanza e hash dei pesi.

---

## 9. File chiave

| File | Ruolo |
|---|---|
| `MASTER_PLAN.md` | Contesto, stato dell'arte, prodotto, decisioni, riferimenti |
| `IMPLEMENTATION_PLAN.md` | Questo documento: fasi e criteri di completamento |
| `pyproject.toml` | Dipendenze di base ed extra `ml` |
| `src/forensicaudio/config.py` | `Mode`, `StageKind`, `CaseConfig`; qui va `Tier` |
| `src/forensicaudio/models.py` | Schemi Pydantic e invariante probatorio |
| `src/forensicaudio/pipeline.py` | Orchestratore e catena di custodia |
| `src/forensicaudio/stages/*.py` | Stadi audio |
| `src/forensicaudio/cli.py` | CLI di sviluppo |
| `scripts/synth_test_audio.py` | Audio sintetico per i test di funzionamento |
| `tests/test_pipeline_smoke.py` | Test end-to-end |

---

## 10. Riferimenti

- Evidenze e fonti: `MASTER_PLAN.md`, sezione 12.
- Censimento dei modelli di maggio 2026: `MASTER_PLAN.md`, Appendice D.
- Kalliope (metodo di esportazione ONNX verificata): `/workspace/kalliope/tools/pocket_export/`.
- Memoria della prima sessione: `/home/dev/.claude/projects/-workspace-nuovo-progetto/memory/` (riflette l'impostazione di maggio 2026).
