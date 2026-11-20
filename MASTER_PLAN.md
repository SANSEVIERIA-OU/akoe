# Master Plan
## Akoe — Trascrizione assistita, adattata al soggetto e al veicolo, per intercettazioni ambientali
### Speaker- and Channel-Adapted Transcription Support for Covert In-Vehicle Recordings: an Experimental Product

Documento di lavoro e base per un paper. Riassume contesto, decisioni, stato dell'arte verificato e metodo.

**Nome.** Fino al 2026-11-20 il progetto si chiamava WHISPER; è stato rinominato **Akoe** (dal greco ἀκοή, "l'udito") per non confonderlo con il modello Whisper di OpenAI che usa al suo interno. Repository: `github.com/SANSEVIERIA-OU/akoe`, sotto SANSEVIERIA License 1.1.

**Revisione 2 — 2026-11-20.** Sostituisce l'impostazione del 2026-05-25 (pulizia generalista + estrazione del parlante bersaglio da miscele di voci). Le ragioni del cambio sono nella sezione 2 e nel log decisioni (sezione 9). Il censimento dei modelli fatto a maggio 2026 è conservato nell'Appendice D.

**Legenda fonti.** † = fonte letta solo in parte (abstract, sintesi automatica o fonte secondaria). Senza simbolo = testo primario letto durante le ricerche del 2026-11-20. Le fonti dell'Appendice D sono quelle raccolte il 2026-05-25.

---

## Abstract

Akoe è un prodotto sperimentale open-source (modello di distribuzione non ancora deciso) che aiuta trascrittori e periti a capire cosa dice una persona registrata da una microspia. Parte dallo scenario più frequente e più trattabile: il soggetto da solo in auto, spesso in soliloquio. Invece di promettere una trascrizione "certa", il sistema combina quattro elementi: (1) un profilo del soggetto costruito da sue registrazioni reali, con un'impronta vocale discriminativa per riconoscere quando parla lui e un profilo timbrico-spettrale per separare la sua voce dal rumore; (2) un profilo acustico del veicolo, misurato o stimato; (3) un trascrittore automatico adattato a quella voce e a quel canale; (4) un risultato fatto di interpretazioni alternative con punteggi onesti, un confronto tra frasi candidate ("ha detto X o Y?") e un'autovalutazione per ogni caso. Il trascrittore umano resta l'autore della trascrizione: lavora prima alla cieca e solo dopo vede le ipotesi della macchina. L'architettura mantiene la separazione tra trattamenti riproducibili e ipotesi AI, la catena di custodia SHA-256 e tre livelli hardware (portatile senza GPU, workstation, cloud sotto controllo dell'utente), con un motore Python e un'interfaccia desktop Tauri. Le scelte si basano su una revisione della letteratura del novembre 2026. Questa mostra margini reali: in condizioni controllate i sistemi automatici battono gli ascoltatori umani, e l'adattamento al parlante dà grandi guadagni. Mostra anche lacune: non esiste nessuno studio su audio reale da microspia in auto.

---

## 1. Motivazione e contesto

### 1.1 Il problema operativo

Le intercettazioni ambientali producono audio compromesso da rumore, riverbero, voci sovrapposte, microfoni economici e lontani, volume variabile. Questo documento si concentra su uno scenario specifico, scelto perché frequente e tecnicamente più trattabile.

**Scenario primario: il soggetto da solo in auto.**
- **Un solo parlante**: niente sovrapposizioni e niente rischio di attribuire al soggetto parole altrui, a patto di verificare che a parlare sia davvero lui (l'auto può essere guidata da altri).
- **Un solo microfono** nascosto, economico, in posizione fissa e nota.
- **Rumore** di motore, rotolamento, vento e ventilazione; spesso radio o musica.
- **Parlato spontaneo e spesso autodiretto (soliloqui)**: probabilmente a bassa voce, frammentario, con articolazione ridotta, spesso in dialetto.
- **Abitacolo accessibile** agli inquirenti, che vi hanno installato la microspia: il canale acustico si può misurare.

Gli scenari con più parlanti o in ambienti chiusi restano obiettivi futuri (sezione 11).

### 1.2 Ipotesi di lavoro (rivista)

L'ipotesi del 2026-05-25 era che il parlato fosse "sotto la soglia di intelligibilità umana ma non sotto quella algoritmica". La revisione la riformula così:

> Con un sistema adattato alla voce del soggetto e al canale acustico del veicolo, la macchina può proporre ipotesi di trascrizione che permettono al trascrittore umano di capire di più e meglio, a patto che l'incertezza sia resa esplicita e che il trascrittore non venga condizionato.

Il sistema non promette di recuperare informazione che non è nel segnale. Dove la voce è coperta dal rumore, un modello può solo attenuare o indovinare, e il prodotto deve dirlo ("incomprensibile", "nessuna ipotesi compatibile").

### 1.3 Cornice normativa italiana

- Le intercettazioni sono regolate dagli artt. 266-271 c.p.p. (riforme: D.Lgs. 216/2017; D.L. 161/2019, conv. L. 7/2020).
- **Art. 268, comma 7 c.p.p.**: la trascrizione delle registrazioni segue "le forme, i modi e le garanzie previsti per l'espletamento delle perizie" (testo letto su Brocardi; da verificare su Normattiva).
- **Cassazione, Ufficio del Massimario, Relazione n. 35/2020** (23/03/2020): la trascrizione serve a rendere il contenuto "più agevolmente consultabile" e non costituisce di per sé prova, che resta la registrazione. Per Akoe questo conta: uno strumento di supporto alla trascrizione rientra in questo quadro, purché non si presenti come fonte di prova.
- La legge non definisce con precisione la trascrizione forense (AISV, Studi 10, 2023). Non esistono linee guida standardizzate per trascrivere le intercettazioni (Meluzzi 2025, preprint). Esiste una proposta di linee guida AISV-OLF del 2019† ed è aperto un dibattito sul carattere descrittivo o interpretativo della trascrizione (Romito 2025†, Camera penale di Catanzaro).
- Secondo la stampa (7/9/2026), la Procura generale di Perugia ha adottato linee guida che ammettono l'AI come ausilio con verifica umana obbligatoria (testo non letto).
- **Verifica legale: decisione del proprietario (2026-11-20).** Il software è destinato alle forze dell'ordine e il proprietario considera soddisfatta la verifica legale (issue #47 chiusa). Restano questioni distinte i termini di licenza dei modelli di terze parti (#46).

### 1.4 Rischi specifici dell'AI in contesto probatorio

- **Allucinazioni del trascrittore automatico.** Circa l'1% delle trascrizioni di Whisper contiene frasi inventate, e il 38% di queste è dannoso (Koenecke et al. 2024†). Con rumore più forte della voce (−4/−2 dB) Whisper inserisce parole inesistenti circa 20 punti percentuali più spesso degli ascoltatori umani (Kim et al. 2024).
- **Parole fantasma e priming.** Chi legge una trascrizione suggerita tende a "sentirla" nell'audio, anche quando è inventata (Fraser 2018†). La sicurezza soggettiva non indica l'accuratezza (Fraser & Kinoshita 2021†).
- **Bias verso le parole attese.** Fornire al trascrittore una lista di nomi e termini del caso dimezza gli errori su quei termini. Ma quando il termine non è stato detto, il sistema tende a inserirlo: errori sulle parole attese fino a +55% e sulle altre fino a +20%, quasi tutti inserimenti (Huber & Waibel 2026b).
- **Italianizzazione del dialetto.** Su napoletano, Whisper non adattato tende a tradurre in italiano invece di trascrivere (Cacioli 2026†).
- **Pulizia che peggiora.** Una riduzione del rumore generica può aumentare gli errori del trascrittore automatico rispetto all'audio originale (Eskimez et al. 2021: 40,1% contro 36,4%).

La risposta architetturale ha cinque parti: separazione tra trattamenti riproducibili e ipotesi AI (`audio_provenance`), ipotesi multiple invece di una trascrizione unica, revisione cieca, autovalutazione per caso, controlli sugli inserimenti (sezione 4).

---

## 2. Evoluzione dell'impostazione (revisione 2026-11-20)

La prima impostazione (maggio 2026) prevedeva una pipeline generalista: pulizia neurale, estrazione del parlante bersaglio (TSE) da miscele di voci, trascrizione. La revisione critica del novembre 2026 ha portato a quattro conclusioni.

1. **I numeri della TSE non si trasferiscono al campo.** Sono misurati su miscele sintetiche di parlato letto e pulito, quasi sempre a due voci (Libri2Mix, WSJ0-2mix). Inoltre la TSE può estrarre la persona sbagliata quando le voci sono simili, ed estrae comunque "qualcosa" anche quando il soggetto non parla. Il rischio è attribuirgli parole altrui.
2. **L'orecchio umano non è sempre migliore della macchina.** Su questo la critica iniziale era stata troppo netta: in esperimenti controllati Whisper batte gli ascoltatori umani (3.1). Su audio forense reale, però, i dati sono quasi assenti.
3. **Lo scenario "soggetto solo in auto" cambia il problema.** Elimina il rischio di attribuzione tra parlanti, salvo verificare chi parla, e rende il canale acustico misurabile.
4. **Il valore sta nell'adattamento al soggetto e nell'incertezza esplicita**, non in una pulizia più aggressiva.

Decisioni conseguenti (dettaglio nella sezione 9):
- obiettivo: un prodotto sperimentale, non un esperimento separato;
- TSE accantonata per lo scenario primario;
- introdotti impronta vocale discriminativa, profilo timbrico, profilo del veicolo, adattamento del trascrittore, ipotesi multiple, autovalutazione e interfaccia Tauri;
- esclusa la clonazione della voce del soggetto.

---

## 3. Stato dell'arte (revisione 2026-11-20)

Ricerche condotte il 2026-11-20 da più ricercatori in parallelo, su fonti primarie. Molte fonti erano a pagamento o bloccate, e il browser interattivo non era disponibile. Dove un numero viene solo da abstract o fonti secondarie è segnato †. I numeri vanno riverificati prima di decisioni operative.

### 3.1 Macchina contro orecchio umano

| Studio | Condizioni | Risultato |
|---|---|---|
| Patman & Chodroff 2024 | 60 ascoltatori; frasi in rumore "speech-shaped" e da pub, a 0 e 8 dB | Whisper large meglio degli umani in tutte le condizioni tranne il rumore da pub, dove pareggia. A 0 dB di rumore da pub sbaglia comunque il 44% delle parole |
| Kim et al. 2024 | Frasi con accento straniero, rumore da −4 a +8 dB | Whisper meglio degli umani; Google, wav2vec 2.0 e HuBERT peggio. Whisper inserisce circa 20 punti percentuali più parole inesistenti a −4/−2 dB |
| Loakes 2024 | Una clip forense reale di 44 s (116 parole), 8 sistemi | Miglior Whisper: 50% di parole sbagliate; altri sistemi 73-98%; sulla clip pulita 4-18% |
| Spille, Kollmeier & Meyer 2018† | Piccolo vocabolario, rumore | Non verificato (testo inaccessibile) |
| Lippmann 1997† | Rassegna, prima del deep learning | Errori delle macchine spesso un ordine di grandezza superiori a quelli umani |

Nessuno studio confronta umani e macchine su registrazioni nascoste in auto.

### 3.2 Trascrizione forense e collaborazione umano-macchina

- **Variabilità umana.** Otto trascrittori sulla stessa registrazione hanno prodotto lo stesso testo solo in 5 battute su 170; accordo sull'attribuzione dei parlanti α = 0,41 (Love & Wright 2021†).
- **Trascrittori "ad hoc".** Le trascrizioni fatte da investigatori che conoscono il caso sono "highly prone to be unreliable". Serve un trascrittore indipendente e formato, con smascheramento sequenziale delle informazioni di contesto (French & Fraser 2018).
- **Bozze automatiche.**
  - Su italiano (corpus KIParla, 11 trascrittori) la bozza di Whisper accelera il lavoro e riduce gli errori per 7 trascrittori su 11, senza un miglioramento sistematico (Simonotti et al. 2026).
  - Evidenziare le parole a bassa confidenza non migliora la correzione (Kuhn et al. 2025).
  - Su audio forense pessimo, Loakes 2024 ritiene più efficiente trascrivere da zero che correggere la bozza.
- **Uso dichiarato.** In letteratura forense la trascrizione automatica è accettabile come bozza verificata da un trascrittore umano, con confidenza per parola (Stoykova, Porter & Beka 2024†).

Implicazione per il prodotto: revisione cieca come impostazione predefinita. Il trascrittore scrive prima, poi vede le ipotesi; nessuna bozza mostrata in partenza.

### 3.3 Acustica dell'abitacolo e pulizia riproducibile

- **Bande dell'intelligibilità.** Pesi d'ottava dello Speech Intelligibility Index (ANSI S3.5-1997): 250 Hz 0,06; 500 Hz 0,17; 1 kHz 0,24; 2 kHz 0,26; 4 kHz 0,21; 8 kHz 0,05. Le ottave 1-4 kHz pesano circa il 72% dell'intelligibilità. Ridurre l'energia sotto i 500 Hz costa poco in comprensibilità.
- **Spettro del rumore in abitacolo.** Nessuna misura verificata trovata, solo fonti commerciali non documentate e in disaccordo tra loro. Va misurato sul veicolo.
- **Cancellazione con riferimento.** Se radio o musica si identificano per impronta acustica e se ne ottiene l'originale, lo si allinea e lo si sottrae con un filtro adattivo (Alexander, Forth & Tunstall 2012, Oxford Wave Research).
  - Gli autori la definiscono "highly effective", ma avvertono che la sottrazione semplice "rarely" basta: in abitacolo il riverbero della sorgente è forte quasi quanto il suono diretto.
  - Mancano numeri indipendenti; gli autori lavorano per un'azienda del settore.
  - Un lavoro del 2026 riporta 30-34 dB di riduzione della correlazione con il riferimento, e avverte del rischio di eliminare contenuto utile se la voce "trapela" nel riferimento (Erkek & Ozcan 2026†).
- **Linee guida SWGDE 20-A-001-2.0 (2025).**
  - I trattamenti devono essere ripetibili e riproducibili "to the extent that they are perceptually consistent", documentati con software, impostazioni e versioni.
  - Gli algoritmi proprietari o non pubblicati vanno testati prima dell'uso.
  - Tra le tecniche elencate ci sono sottrazione spettrale, cancellazione con canale di riferimento e "deep learning filters".
  - Le misure oggettive "may not be effective in forensic audio".
- **Altre linee guida.** ENFSI ha un manuale sull'autenticità (FSA-BPM-002, 2022) ma nessuno sull'enhancement. Non è stata trovata nessuna linea guida specifica sull'AI.
- **Trascrizione in auto (benchmark).** Anche con parlanti collaborativi e microfoni montati nell'abitacolo, i sistemi sbagliano molto:
  - ICMC-ASR (cinese, 4 microfoni): miglior CER 13,2% contro 26,2% della baseline†;
  - AISHELL-5: baseline CER 26,1%;
  - CU-Move (anni 2000): WER 30-65%.
  - I finestrini aperti pesano più della velocità (Weng et al. 2016).
- **Lacuna.** Nessuna prova verificata che i filtri classici (sottrazione spettrale, Wiener) migliorino l'intelligibilità per ascoltatori umani nel rumore d'auto.

### 3.4 Adattamento al parlante e al canale

| Tecnica | Evidenza | Limite |
|---|---|---|
| Adattamento del trascrittore al parlante | 71% del miglioramento con 5 min di dati (Shor et al. 2019†); 63% dei parlanti raggiunge l'obiettivo con 3-4 min, 79% con 18-20 (Tobin & Tomanek 2021†); WER mediana 4,6% contro 31% (Green et al. 2021†); un parlante disartrico con Whisper-large-v3 passa da 128% a 15,8% con 1,4 h, e LoRA di rango 8 è peggiore del fine-tuning completo del 15-39% relativo (Huber & Waibel 2026a); con microfono lontano 1 minuto di adattamento porta da 24,4% a 19,4% (van Dalen et al. 2025) | Quasi tutto su parlato atipico registrato pulito; nessuna curva minuti/errore nel rumore reale |
| Parlato sintetico come integrazione | 21 min reali: 21,9%; con aggiunta di sintetico: 18,3% (Mihajlik et al. 2025) | Escluso per Akoe (4.2) |
| Riduzione del rumore personalizzata | Errori del trascrittore: 36,4% sull'originale, 40,1% con riduzione generica, 29,6% e 25,8% con modelli personalizzati da circa 60 s di campione (Eskimez et al. 2021) | Sopprime anche parte della voce; altri studi trovano peggioramenti rispetto all'originale (arXiv 2212.00406) |
| Simulazione del canale reale | Riverbero misurato nello stesso ambiente: 12,4% di errori, contro 77,2% addestrando su voce pulita e 16,5% con riverbero simulato (IR-GAN 2020); in abitacolo CER da 17,4% a 16,6% (CabinSep 2025) | Nessuno studio in auto che confronti canale misurato e generico; in CHiME-3 i sistemi addestrati su dati simulati peggiorano rispetto all'originale (arXiv 2205.01751) |
| Vocabolario del caso | Errori sulle parole chiave più che dimezzati (Liu et al. 2025). ROXANNE (progetto UE per le polizie, dati simulati): F1 sui nomi da 15,4 a 20,6 in inglese e da 43,3 a 54,6 in tedesco | Inserimenti quando la parola non c'è (1.4); senza filtri, gli errori sulle altre parole passano dall'1,9% al 21,2% (Liu et al. 2025) |

#### 3.4.1 Pulizia prima del trascrittore automatico: di solito peggiora

| Studio | Trascrittore e condizioni | Audio rumoroso | Audio pulito | Pulito + quota di originale |
|---|---|---|---|---|
| Chondhekar et al. 2025 | Whisper large-v3, rumore a 10 dB (metrica semWER) | 8,8% | 25,8% (MetricGAN+; peggiore in tutte le 40 configurazioni) | — |
| Wang et al. 2024 | Whisper large-v3, LibriSpeech + rumore DNS | 15,4% | 27,2% (CMGAN) | 15,0% |
| Cui et al. 2025 | Whisper large-v3 | 4,66% | 6,19% (FRCRN) | 4,25% |
| Sato et al. 2025 (NTT) | Whisper medium | 9,3% | 12,2% | 9,1% (10% di originale) |
| Huo et al. 2026 | Whisper large-v3, VoiceBank+DEMAND | 6,54% | 6,44% (FRCRN) | 5,57% (correzione applicata al 25%) |

- **Rimescolare una quota dell'originale** ("observation adding") recupera il danno e a volte migliora di poco. Su più microfoni può però peggiorare (SuperM2M: un microfono 6,02 → 5,80; sei microfoni 4,04 → 4,24).
- **Fusione appresa** di audio rumoroso e pulito: aiuta HuBERT (CHiME-4 da 8,6% a 5,7%, mentre anteporre la pulizia non dava guadagni; FAT-HuBERT 2023). Non è testata su Whisper.
- **Dopo il fine-tuning la pulizia non aggiunge nulla**: "(generative) SE has no additional benefit to fine-tuning" (de Groot et al. 2025, parlato disartrico).
- **Fine-tuning su dati rumorosi**: risultati variabili. Sundanese a SNR negativo da 79,9% a 41,4%, ma giavanese da 100,8% a 115,0% (Pranida et al. 2025). Canary su audio telefonico: CER da 23,3% a 9,0% con simulazione di codec mu-law e rumore (Boonpramuk et al. 2026).
- **Lacuna**: nessun confronto controllato "fine-tuning con risposta all'impulso e rumore" contro "pulizia davanti al trascrittore" su Whisper o Parakeet.

**Conclusione per Akoe.** La conoscenza della voce e del canale va dentro il trascrittore automatico (adattamento), non in un filtro davanti. La pulizia davanti al trascrittore si usa solo rimescolata con l'originale, e solo se l'autovalutazione del caso mostra che aiuta.

#### 3.4.2 Ricette di simulazione

- **Riverbero e rumore sono gli ingredienti principali.** Su Aurora-4: addestrando solo su voce pulita 26,98% di errori, con tutte le aumentazioni 10,55%, togliendo il riverbero 11,59% (Oglic et al. 2022).
- **Il rumore reale dell'ambiente conta quanto la risposta all'impulso.** Con rumore della stanza corrispondente, risposte all'impulso artificiali e reali danno quasi lo stesso risultato: AMI a microfono singolo da 72,7% a 47,1% (artificiali), 47,4% (reali), 46,1% (combinate) (Szöke et al. 2019).
- **Allineamento.** Risposte all'impulso reali 11,80% contro simulate 15,56%; una convoluzione disallineata aumenta gli errori (Tang et al.). Il ritardo va compensato.
- **Codec.** La simulazione di codec (AMR-NB, Ogg) ha aiutato in campo lontano (29,2% → 17,6%) ma ha peggiorato un altro test (12,0% → 13,7%) (Balam et al. 2020). Va usata solo se la catena di trasmissione della microspia usa davvero un codec.
- **Risposta del microfono**: nessuna evidenza per la trascrizione automatica.

#### 3.4.3 Tecniche di adattamento, modelli e hardware

- **LoRA o fine-tuning completo.** Il completo è in genere pari o leggermente migliore:
  - Whisper-small su bambini: completo 9,3%, LoRA 9,6%, adattatori 9,3% (Fan et al. 2024);
  - Whisper large-v3 su svizzero tedesco: 25,6% contro 26,3%;
  - singolo parlante disartrico: LoRA di rango 8 peggiore del 15-39% relativo (Huber & Waibel 2026a);
  - un caso su body-cam delle forze dell'ordine con 42 clip va in senso opposto (LoRA 0,373 contro completo 0,587), ma con un fine-tuning completo debole (Senthil & Fokoué 2026).
- **Adattamento al momento della trascrizione** (test-time adaptation). Whisper-base, solo LayerNorm, 10 tipi di rumore a 10 dB: da 22,5% a 19,2% (arXiv 2605.08186). SUTA è peggiorato in 3 rumori su 8 (arXiv 2603.05231). Costoso; opzione futura.
- **Modelli italiani adattabili:**
  - Parakeet-TDT 0.6B v3: CC-BY-4.0, 0,6 miliardi di parametri, FLEURS italiano circa 3,0%;
  - Canary-1B-v2: CC-BY-4.0, 978 milioni;
  - Whisper large-v3: MIT secondo il repository OpenAI, 1,55 miliardi.
  - Classifica Open ASR (italiano): Parakeet 4,81%, Canary 4,88%.
- **Hardware:**
  - non esistono cifre ufficiali di memoria per LoRA sui modelli grandi;
  - esecuzioni misurate: GPU da portatile con 12 GB (LoRA su large-v3-turbo, progetto non revisionato) e A100 con 20 GB;
  - la stima "sotto 8 GB" per large-v2 è di fonte secondaria;
  - fine-tuning senza GPU: nessun tempo misurato. Su GPU integrata, Whisper-tiny ha richiesto 17 min per 1.000 passi su 7,7 h di audio, e small 55 min (blog).

### 3.5 Analisi del timbro e separazione della voce dal rumore

#### 3.5.1 Pulizia informata dal parlante

| Famiglia | Evidenza | Parametri ispezionabili |
|---|---|---|
| Dizionari NMF (voce e rumore) | Per il rumore, un modello piccolo specifico per quel tipo batte uno grande generico. Le basi del parlante servono a separare voci, non sono indispensabili per ridurre il rumore (Mohammadiha, Smaragdis & Leijon 2013; rumori di fabbrica, brusio, traffico, non auto). Nella tesi KTH 2013 le basi del parlante danno +0,45 PESQ rispetto a Wiener a 0 dB (una sola condizione, solo voci maschili) | Sì (basi) |
| Codebook di inviluppi spettrali | Codebook adattato al parlante da circa 40 s, come post-filtro: migliora ad alto SNR (a 20 dB distorsione spettrale da 5,07 a 3,22), leggermente peggio in alcuni casi a −5 dB; 2 soli parlanti (Chidambar & Hanumantha Rao Naidu 2024) | Sì (voci del codebook) |
| Modelli armonici (F0) | DNN con modello armonico: PESQ 1,58 contro 1,43 a −10 dB (Ouyang et al. 2018). Nessun modello di F0 specifico del parlante testato | In parte |
| Neurali specializzati sul parlante | 10 h di un parlante contro un generalista da 100 h: +0,7-1,3 dB SI-SDR, ESTOI +0,014-0,034 in nove architetture (Leer et al. 2026, preprint). Personalizzazione anche con pochi secondi di voce pulita (Sivaraman & Kim 2022) | No |
| Test con ascoltatori | Unico confronto con ascoltatori, su portatori di impianto cocleare: modello addestrato su 8 min del parlante contro 15 min di altri. Miglioramento della soglia di comprensione: +2,3 contro +1,4 dB (rumore stazionario), +6,4 contro +5,4 (ICRA), +2,0 contro +0,4 (brusio di voci) (Goehring et al. 2017). Con parlanti nuovi, i modelli specializzati danno guadagni in alcune condizioni e perdite in altre (Kolbæk et al. 2017) | — |

**Lacuna:** nessuno studio prova insieme un parlante noto, rumore d'auto, SNR basso e misure di intelligibilità o WER.

**Conclusione per Akoe.** La pulizia informata dal parlante dà guadagni moderati e reali soprattutto per chi ascolta. È una versione d'ascolto per il trascrittore umano, etichettata come neurale; non è l'ingresso principale del trascrittore automatico (3.4.1).

#### 3.5.2 Misure del timbro

- **F0 di lungo periodo:** si stabilizza dopo circa 10 s di parlato (media 9,76 s); il parlato spontaneo è il più variabile (Arantes, Eriksson & Gutzeit 2017†).
- **Formanti:**
  - stabili tra un giorno e l'altro; nello stesso giorno F1 sale di 13 Hz e F0 di 9,4 Hz, sotto la soglia di discriminazione (Heald & Nusbaum 2015, 8 parlanti);
  - la misura manuale di F1 sbaglia fino a 118 Hz su vocali sintetiche (Shadle, Nam & Whalen 2016). Serve una misura automatica con parametri registrati, solo su tratti abbastanza puliti.
- **Canale:** codec e ricampionamento alterano F1 (spostamento medio di circa il 29% ricampionando 16 → 8 → 16 kHz, in uno studio sui deepfake; Sukhdeve et al. 2026). Servono profili separati per canale.
- **Spettro medio di lungo periodo:** nessuna durata minima verificata in letteratura forense. Lo spettro medio del parlato è simile tra 12 lingue (Byrne, Dillon & Tran 1994†).

#### 3.5.3 Sforzo vocale, effetto Lombard, soliloqui

- **Lo sforzo vocale cambia la voce abbastanza da ingannare i sistemi di riconoscimento del parlante:**
  - sussurrato contro normale: EER 26,6% (x-vector) contro 6,8% (ReDimNet-B6) e 5,3% con il metodo proposto (Gołębiowska & Syga 2026);
  - gridato contro normale: EER 21,8%, che scende a 17,7% con compensazione (López-Espejo et al. 2023);
  - con il rumore l'EER passa da 7,8% a 24,5% (Gołębiowska & Syga 2026).
- **Effetto Lombard in auto:** F0 e intensità salgono con il rumore di guida (solo da estratti di ricerca); nessuna quantificazione specifica verificata.
- **Soliloqui:** uno studio del 2025 rileva il parlato autodiretto (F1 macro 0,84, 25 partecipanti) e lo descrive come "a basso volume" e "borbottato", senza misure. Nessun dato di trascrizione automatica sul parlato autodiretto (Lee et al. 2025†).
- **Italiano spontaneo:** su KIParla Whisper appiattisce le autocorrezioni e omette le parole brevi (Simonotti et al. 2026†).

**Conclusione per Akoe.** Il profilo timbrico va costruito **per condizione** (canale: telefono, stanza, auto; sforzo: normale, basso, alzato). Il materiale di riferimento migliore sono le registrazioni precedenti del soggetto **nella stessa auto**, una volta verificate dal trascrittore: stesso canale, stesso effetto Lombard, stesso modo di parlare da solo.

### 3.6 Interpretazioni multiple e incertezza

#### 3.6.1 Generare alternative

- **Beam search di Whisper:** le ipotesi N-best hanno una "very limited diversity". Con le 5 migliori, l'errore oracolo su CHiME-4 scende da 11,1% a 9,1% (HyPoradise, Chen et al. 2023).
- **Campionamento + MBR** (scelta del candidato più "centrale" tra molti campioni). Con Whisper-large-v3 a 0 dB di rumore MUSAN gli errori scendono da 8,1% a 5,7%; sul parlato conversazionale AMI salgono invece da 38,0% a 56,8% (Jinnai 2025).
- **Combinazione di sistemi diversi (ROVER):** nel 1997 da 44,9% a 39,4%; Kaldi + XLS-R circa 14% relativo in meno, circa 20% su dati rumorosi (Parikh et al. 2024†).
- **Punteggi interni CTC:** nessuna delle 11 strategie batte la decodifica semplice (Novosad 2026†).
- **Lacuna:** nessuna fonte riporta, per Whisper o Parakeet nel rumore, quante volte la frase corretta è tra le prime N.

#### 3.6.2 Confidenza

- **Confidenza basata sull'entropia (NeMo, entropia di Tsallis con α = 0,33):** individua le parole sbagliate fino a 2 volte meglio (CTC) e 4 volte meglio (RNN-T) della probabilità massima (Laptev & Ginsburg 2022†). È disponibile in NeMo (`confidence_cfg`), anche per i modelli TDT.
- **Whisper nel rumore:** una temperatura selettiva riduce l'errore di calibrazione da 0,086 a 0,036 (tra −18 e −5 dB), ma la separazione tra parole giuste e sbagliate resta debole (EER circa 25%) (Huo et al. 2025).
- **C-Whisper:** un decoder rifinito per stimare la confidenza raggiunge su CHiME-6 un'AUC di 0,898 contro 0,694 (Aggarwal et al. 2025).
- **Effetto sugli umani:** evidenziare le parole a bassa confidenza non aiuta i correttori (Kuhn et al. 2025).

#### 3.6.3 Allucinazioni

- **Su ingressi senza voce** Whisper large-v3 produce testo nel 40,3% dei casi. Una lista di allucinazioni tipiche combinata con il rilevamento della voce porta l'errore su un test aumentato da 104,8% a 6,5% (Barański et al. 2025).
- **Disaccordo tra sette modelli** (sovrapposizione del vocabolario): AUC 81% nel rilevare allucinazioni (HALAS, Barański et al. 2026†).
- **Intervento sull'encoder:** allucinazioni su non-voce da 86,9% a 27,3% per large-v3 (Aparin et al. 2026†).

#### 3.6.4 Punteggio di frasi candidate ("ha detto X o Y?")

- **Strumenti:**
  - `torchaudio.functional.forced_align`: modelli CTC, un elemento per volta; secondo il tutorial 2.8 è deprecato e verrà rimosso nella 2.9;
  - NeMo Forced Aligner: "only CTC models, or Hybrid CTC-Transducer models (in CTC mode)";
  - CTC-segmentation (SpeechBrain).
  - Nessuna valutazione nel rumore per nessuno dei tre.
- **Whisper** non ha un'API per valutare un testo dato. Tecnicamente si può calcolare la verosimiglianza di un testo forzandolo nel decoder, ma nelle fonti lette non è documentato né valutato.
- **L'addestramento consapevole del rumore aiuta la verifica di parole chiave:** a −5 dB il richiamo passa da 41,6% a 56,6% (NTC-KWS 2025).
- **Insiemi di candidate piccoli:** la correlazione tra punteggio CTC ed errore reale si indebolisce al crescere dei candidati (−0,57 con 4, −0,27 con 128; Novosad 2026†).
- **Cautela di Loakes 2022:** su audio scadente l'allineamento può "sembrare funzionare" su testo sbagliato.
- **Lacuna:** nessuna validazione forense del punteggio di candidate nel rumore. Le soglie vanno tarate per caso, nell'autovalutazione, confrontando la frase giusta con frasi alterate.

#### 3.6.5 Presentare le alternative agli umani

- Nessuno studio controllato misura l'accuratezza dei trascrittori a cui si mostrano liste N-best o confusion network.
- Un "copilota" che propone la sola ipotesi migliore riduce gli errori del 15% relativo (HTEC, Sun et al. 2023).
- L'effetto delle alternative sul trascrittore umano resta da osservare nel prodotto.

### 3.7 Impronta vocale e attribuzione

- **Che modello serve.** Un'impronta per *riconoscere* il parlante va prodotta da un modello di verifica del parlatore (famiglie ECAPA-TDNN, ResNet di WeSpeaker, CAM++), addestrato perché registrazioni della stessa persona in condizioni diverse risultino vicine. La scelta del modello e della licenza è ancora da fare (4.2).
- **Cosa non va confuso.** Lo "stato della voce" di Kalliope è una rappresentazione per la sintesi: imita tutto ciò che sente, compresi rumore ed eco, e non misura l'identità. Non è adatto a questo scopo.
- **Attribuzione come prova.** Attribuire una voce a una persona come prova è materia di comparazione forense del parlatore. Le linee guida ENFSI chiedono validazione su dati tipici del caso e considerano insufficienti le cifre dei fornitori (Drygajlo et al. 2015). Il metodo spettrografico "a impronta" non è scientifico, e IAFPA (2007) ne sconsiglia l'uso nei casi reali (Morrison 2009).
- **Ruolo in Akoe.** L'impronta è uno strumento operativo (smistare, selezionare, avvisare), mostrato come punteggio e mai come identificazione.

### 3.8 Fonetica forense: spettrogrammi e frasi contese

- **Lettura degli spettrogrammi.** Un esperto con circa 2.000 ore di pratica concordava all'80-90% con fonetisti addestrati sull'identificazione dei fonemi (Cole et al. 1978†) e riconosceva il 91% delle parole aiutandosi col contesto (Zue & Cole 1979†). Erano registrazioni di parlato letto e pulito.
- **Escludere più che stabilire.** L'analisi acustica è "generally more effective in ruling out" una trascrizione proposta che nello stabilire il contenuto (Fraser 2021).
- **Soglia proposta.** Paoloni (AISV) propone di trascrivere un passaggio solo se nello spettrogramma si riconoscono almeno il 90% delle vocali e il 50% delle consonanti. Nello stesso lavoro, 12 esperti giudicavano "possibili" frasi candidate su segnali incompatibili (50% su arabo, 10% su italiano, 20% su sintetico). La soglia non è validata.
- **Prassi italiana.** In un caso riportato gli esperti esaminano prima lo spettrogramma e poi ascoltano, per limitare il condizionamento (Cenceschi & Meluzzi 2023).
- **Confronto tra candidate con modelli acustici.**
  - Non è stato trovato nessuno studio forense che usi l'allineamento forzato per confrontare trascrizioni concorrenti.
  - Esiste il calcolo di rapporti di verosimiglianza per singole parole contese a partire da misure acustiche, con l'avvertenza che con pochi dati i valori estremi sono irrealistici (Morrison, Lindh & Curran 2014†).
  - L'allineamento forzato su una registrazione scadente di 44 s ha prodotto più parole mal allineate che corrette, e "systems can appear to work on transcription data that is simply wrong" (Loakes 2022†).

### 3.9 Dialetti

- **Napoletano.** Whisper non adattato sbaglia circa l'80% delle parole (small → medium: 81,2% → 78,5%). Passando al modello più grande l'errore di italianizzazione cresce del 44,5% relativo. L'adattamento LoRA lo riduce del 55-69% (Cacioli 2026†; corpus di 591 clip, 4 parlanti).
- **Siciliano e calabrese.** Nessun corpus o modello verificato; i corpus Primula e SoPhISM sono solo citati.
- **Implicazione.** Il materiale reale del soggetto, trascritto nel suo dialetto, è la risorsa principale. Le trascrizioni esistenti vanno controllate, per non insegnare al modello a italianizzare.

### 3.11 Modelli migliori disponibili (ricerca del 2026-11-20)

Criterio del proprietario: il migliore disponibile al mondo, che giri su macchine controllate. Il catalogo per livello è in `docs/catalogo_modelli.md`. Sintesi:
- **Trascrizione italiana (pesi aperti).**
  - Nemotron 3.5 Transcribe (NVIDIA, ottobre 2026): FLEURS-it 1,39; media della classifica 3,61; 67 GB, solo cloud.
  - Cohere Transcribe 03-2026: 2B, Apache-2.0, media 3,44, gira anche su CPU.
  - Qwen3-ASR-1.7B: FLEURS-it 2,41, il più robusto al rumore nel test URGENT 2024 (inglese, 9,93 contro 13,73 di Whisper large-v3).
  - omniASR-LLM-7B e Voxtral Small 24B (media 3,75); Parakeet-TDT 0.6B v3 (FLEURS-it 3,00); Whisper large-v3 (media 4,45).
  - I modelli chiusi (ElevenLabs Scribe v2, Azure) hanno numeri migliori ma sono esclusi dal vincolo. Nessuna misura italiana in rumore o in auto per nessuno.
- **Impronta vocale.** ReDimNet2-B6 (MIT, 12,3M, Vox1-O 0,23–0,29%); w2v-BERT 2.0 SV (0,12%, ~600M). Nei test sullo sforzo vocale la famiglia ReDimNet è la più robusta (Lombard: EER 0,50/0,89 contro 1,68/2,50 di ECAPA-TDNN; sussurro: 6,77 contro 10,49). Codec a bitrate molto basso degradano tutti i modelli (ECAPA-TDNN da 0,86 a 11,75 con Opus 1,5 kbps). Ricetta di punteggio WeSpeaker: AS-norm e QMF.
- **CTC italiano.** omniASR CTC 1B/3B v2 (FLEURS-it 3,9/1,3% su un campione minimo); wav2vec2-xls-r-1b-italian (CV8 9,04/2,20).
- **Pulizia d'ascolto.** UniPASE (MIT) ha la minore distorsione del contenuto, misurata come dWER; i generativi puri mostrano molte allucinazioni (URGENT 2025: un sistema generativo con MOS alto ma ESTOI 0,53 contro 0,82). GTCRN e DeepFilterNet per la CPU.
- **VAD.** Silero VAD 6.2.3 (MIT); FireRedVAD (Apache-2.0). Nessun benchmark in rumore d'auto.
- **Limiti.** Le classifiche italiane sono medie su dataset discussi e snapshot diversi; le misure CPU sono di terze parti. Va tutto rimisurato nell'autovalutazione e sul banco di prova.

### 3.10 Modelli candidati (censimento maggio 2026)

Le tabelle di pulizia, TSE e trascrizione italiana del 2026-05-25 sono nell'Appendice D. Restano valide come elenco di candidati, con due note:
- la TSE è fuori dallo scenario primario;
- per la trascrizione la scelta va rifatta in base all'adattabilità (fine-tuning, LoRA) e alla capacità di produrre alternative e confidenza, non solo al WER su dati puliti.

---

## 4. Il prodotto

### 4.1 Chi lo usa e come

**Utenti:** trascrittori e periti che lavorano sulle intercettazioni; investigatori, per la consultazione.

**Flusso per caso**, tutto su macchine controllate:
1. Apertura del caso: import delle registrazioni, copia di lavoro, SHA-256 (esiste già).
2. Profilo del veicolo (4.3).
3. Profilo del soggetto (4.2).
4. Pulizia (4.4).
5. Adattamento del trascrittore (4.5).
6. Trascrizione con interpretazioni multiple (4.6).
7. Revisione umana cieca ed esportazione (4.8).

L'autovalutazione (4.7) accompagna ogni caso.

### 4.2 Profilo del soggetto

Costruito solo da registrazioni reali del soggetto (altre intercettazioni, telefonate, interrogatori), con le trascrizioni già esistenti quando ci sono.

| Componente | Serve a | Come si ottiene |
|---|---|---|
| **Impronta vocale** | Riconoscere se a parlare è il soggetto; selezionare il suo parlato in ore di altro materiale; avvisare quando la somiglianza cala | Modello di verifica del parlatore; punteggi calibrati sul materiale del caso |
| **Profilo timbrico-spettrale** | Separare la sua voce dal rumore proteggendo le sue bande; tarare i filtri | Misure acustiche (spettro medio, F0, formanti) e modelli della sua voce; metodo in 3.5 |
| **Dati di adattamento** | Specializzare il trascrittore | Suo parlato e relative trascrizioni, fatti passare attraverso il canale del veicolo (4.3) |
| **Lessico del caso** | Nomi, luoghi, gergo | Da trascrizioni precedenti; uso controllato (4.5) |

Una quota del materiale trascritto viene messa da parte e non usata per l'addestramento: serve all'autovalutazione (4.7).

**Modello del soggetto multi-contesto (idea guida dell'utente, 2026-11-20).** L'obiettivo è un modello che impari il timbro e il modo di parlare del soggetto da tutte le occasioni disponibili (telefonate, interrogatori, ambienti chiusi, auto) e che lo riconosca e ne interpreti le parole indipendentemente dal contesto della registrazione. In pratica:
- **Impronta vocale** costruita da campioni di tutte le condizioni, non da una sola; il punteggio resta riferito alla condizione più vicina.
- **Adattatore del trascrittore** addestrato su tutto il materiale verificato del soggetto, in tutte le condizioni. Si aggiungono simulazioni di canali diversi (abitacolo misurato, stanze, telefono), codec variabili e sforzi vocali diversi. È l'addestramento "multi-condizione", la ricetta che in letteratura dà la robustezza maggiore (3.4.2).
- **Specializzazione leggera opzionale** per una condizione specifica (per esempio quell'auto), sopra il modello del soggetto, solo se l'autovalutazione mostra che serve.
- **Limite dichiarato:** "indipendentemente dal contesto" è un obiettivo, non una garanzia. L'autovalutazione misura le prestazioni **per condizione** e mostra dove il modello regge e dove no.

**Profili per condizione.** Lo sforzo vocale e il canale cambiano la voce abbastanza da ingannare i sistemi di riconoscimento del parlante (3.5.3). Impronta e profilo timbrico vanno quindi tenuti separati per condizione (telefono, stanza, auto; voce normale, bassa, alzata), e ogni punteggio dichiara rispetto a quale condizione è calcolato.

**Ciclo di verifica.** Il materiale più prezioso sono le registrazioni precedenti del soggetto nella stessa auto: stesso canale, stesso effetto Lombard, stesso modo di parlare da solo. Ogni tratto che il trascrittore marca come "certo" entra nel profilo e nei dati di adattamento del caso, con la sua versione. I tratti incerti non vi entrano mai, per evitare che il modello impari dagli errori.

**Esclusa la clonazione della voce del soggetto** (per esempio con Pocket TTS di Kyutai, usato in Kalliope). Motivi:
1. Il clone non contiene informazione sul timbro in più rispetto alla registrazione da cui nasce (in Kalliope, 12 secondi). I dettagli fini sono ricostruiti dal modello, quindi tarare filtri sul clone è un ragionamento circolare.
2. I termini d'uso dei pesi con clonazione vietano di imitare una voce senza il consenso esplicito e lecito del titolare. È la lettura riportata nel README di Kalliope; va verificata legalmente.
3. Un sistema investigativo capace di sintetizzare la voce dell'indagato offre un argomento alla difesa.
4. Il sintetico riproduce il timbro ma non il dialetto né il modo di parlare.

Le coppie pulito/rumoroso che servono a tarare i filtri si ottengono facendo passare registrazioni reali e pulite del soggetto attraverso il canale misurato del veicolo.

### 4.3 Profilo del veicolo

- **Misurato (preferibile).** Durante l'installazione della microspia o in un accesso successivo:
  - risposta all'impulso dal posto di guida (ed eventualmente dal passeggero) al microfono;
  - registrazioni del rumore a motore spento, al minimo, in città, in extraurbano e in autostrada, con finestrini chiusi e aperti, ventilazione accesa e spenta.
  - Procedura in un runbook dedicato.
- **Stimato (ripiego).** Profilo del rumore ricavato da tutti i tratti senza voce della registrazione; armoniche del motore tracciate nel tempo.
- **Usi.** Filtri riproducibili; simulazione del canale per l'addestramento e l'autovalutazione; stima del rapporto segnale/rumore per segmento.

**Doppia registrazione e codec variabili (informazione dell'utente, 2026-11-20).** La microspia registra in locale e trasmette, e i codec di trasmissione possono variare. Conseguenze:
- **Due reperti per lo stesso evento**, ciascuno con hash e catena di custodia propri. Quando c'è, la registrazione locale (di solito di qualità migliore) è l'ingresso predefinito della trascrizione; quella trasmessa resta disponibile e confrontabile.
- **Misura del canale di trasmissione.** Allineando registrazione locale e trasmessa dello stesso evento si stima con precisione cosa fa la trasmissione (banda, codec, perdite) e si ottengono coppie "migliore/peggiore" reali, utili per l'adattamento e l'autovalutazione.
- **Codec variabili.** Il codec di ogni file si rileva quando i metadati lo permettono, e si registra come condizione. In addestramento si simula un insieme di codec (G.711, MP3, Opus, Vorbis; AMR e GSM se è disponibile un encoder), così il modello del soggetto non dipende da un codec specifico.

### 4.4 Pulizia

Due strati, mai mescolati in un unico artefatto.

- **Riproducibile (`DETERMINISTIC`).**
  - filtraggio per bande, motivato dall'SII e dal profilo spettrale del soggetto;
  - riduzione del rumore con profilo stimato sull'intera registrazione;
  - attenuazione delle armoniche del motore;
  - cancellazione di radio e musica con riferimento, quando l'originale è disponibile.
  - Parametri e versioni nel log, in linea con SWGDE.
- **Neurale (`NEURAL`).** Pulizia personalizzata sulla voce del soggetto, solo come ipotesi. Va misurata caso per caso, perché può peggiorare il trascrittore automatico.

**Cosa va al trascrittore automatico.** Per impostazione predefinita l'audio originale (eventualmente dopo la cancellazione della radio e filtri lievi). In alternativa, una versione pulita **rimescolata** con una quota dell'originale. Quale variante usare e con quale quota si decide per caso con l'autovalutazione, perché la pulizia davanti a Whisper di solito peggiora (3.4.1). La pulizia personalizzata serve soprattutto all'ascolto del trascrittore umano (3.5.1).

### 4.5 Adattamento del trascrittore

- **È qui che va la conoscenza del timbro.** Le evidenze indicano che adattare il trascrittore alla voce e al canale rende più che filtrare l'audio prima (3.4.1).
- **Modelli di base proposti:**
  - Parakeet-TDT 0.6B v3 come trascrittore principale: WER italiano basso, confidenza basata sull'entropia in NeMo, dimensioni adatte alla CPU;
  - Whisper large-v3 (o turbo) come seconda famiglia: errori diversi e strumenti LoRA diffusi;
  - un modello CTC italiano per l'allineamento e il punteggio delle frasi candidate.
  - Da verificare: se Parakeet-TDT v3 ha una testa CTC usabile dal NeMo Forced Aligner.
- **Dati:** il parlato reale del soggetto, convoluto con la risposta all'impulso misurata e compensato nel ritardo, mescolato con il rumore registrato di quell'auto a vari rapporti segnale/rumore. Il rumore reale dell'abitacolo conta quanto la risposta all'impulso (3.4.2). Simulazione di codec solo se la microspia trasmette compresso.
- **Tecnica:** LoRA come impostazione predefinita, perché portabile e vicina al fine-tuning completo; completo come opzione su workstation o cloud quando l'autovalutazione mostra che conviene (3.4.3).
- **Risultato:** un adattatore leggero (pochi MB) o un modello completo, versionato e con SHA-256. Principio: **si addestra dove c'è potenza, si usa ovunque**.
- **Lessico del caso:** si usa solo con tre garanzie:
  - controllo degli inserimenti (test con parole assenti, nell'autovalutazione);
  - versione registrata nel log;
  - quando le ipotesi "con lessico" e "senza lessico" divergono, si mostrano entrambe.

### 4.6 Trascrizione con interpretazioni multiple

- **Per ogni tratto:** più ipotesi con punteggio, dal modello generico e da quello adattato, con i disaccordi evidenziati.
- **Per ogni parola:** alternative e confidenza calibrata; la calibrazione è misurata nell'autovalutazione.
- **"Ha detto X o Y?":** il trascrittore scrive frasi candidate e il sistema restituisce quanto ciascuna è compatibile con l'audio. Può rispondere che nessuna lo è, in linea con l'idea che l'analisi acustica serva soprattutto a escludere (3.8).
- **"Incomprensibile":** etichetta esplicita quando nessuna ipotesi supera la soglia.

**Metodo (3.6):**
- **Fonti di alternative interpretabili:** modello adattato contro generico; Parakeet contro Whisper; con lessico contro senza; audio originale contro rimescolato. Si aggiungono il campionamento con scelta MBR e le N migliori ipotesi. Ogni alternativa dichiara da dove viene ("il modello adattato sente X, quello generico Y").
- **Confidenza** basata sull'entropia (NeMo), mostrata per fasce e calibrata nell'autovalutazione, non come probabilità grezza.
- **Guardia contro le allucinazioni:** testo prodotto su tratti senza voce, frasi tipiche di allucinazione, disaccordo forte tra modelli. Tutto segnalato, mai scartato in silenzio.
- **"X o Y?":** punteggio per allineamento forzato (modello CTC) e verosimiglianza nel decoder (Whisper), su pochi candidati alla volta, con soglie tarate nell'autovalutazione del caso.

### 4.7 Autovalutazione per caso

Il prodotto misura sé stesso su ogni caso, invece di affidarsi a un esperimento separato.
- **Come.** Il materiale messo da parte (4.2) viene degradato con il canale e il rumore del veicolo, al rapporto segnale/rumore stimato per il caso.
- **Cosa misura:**
  - errori per parola del modello generico e di quello adattato;
  - copertura: quante volte la frase giusta è tra le alternative proposte;
  - calibrazione della confidenza;
  - tasso di inserimento del lessico su parole assenti;
  - affidabilità dell'impronta vocale sul materiale del caso.
- **Come si presenta.** Il risultato accompagna la trascrizione: "su materiale simile, in queste condizioni, il sistema sbaglia circa X parole su 100".
- **Limite dichiarato.** Il materiale messo da parte non è identico ai soliloqui in auto (sforzo vocale, contesto). L'autovalutazione è una stima, non una garanzia.

### 4.8 Revisione umana

- **Cieca per impostazione predefinita:** il trascrittore trascrive senza vedere ipotesi, poi le confronta (smascheramento sequenziale, French & Fraser 2018).
- **Spettrogramma e forma d'onda** calcolati dal motore, con parametri registrati, per la riproducibilità.
- **Ascolto alternato** di originale e versioni pulite, sempre etichettate.
- **Esportazione per la perizia:** il testo del trascrittore, con per ogni parola l'origine (umano; umano confermato dalla macchina; ipotesi della macchina accettata), le alternative scartate e i riferimenti al log.

### 4.9 Modalità e invarianti probatori

Restano le modalità FORENSIC / INVESTIGATIVE / BOTH e l'invariante `audio_provenance`. Rispetto a maggio 2026 cambiano due cose.

**Neurale in modalità forense.** SWGDE ammette anche filtri neurali, se testati e documentati: negli USA la regola "forense = solo deterministico" è più restrittiva del necessario. Per il contesto italiano la si mantiene come impostazione prudente, da rivedere con periti e legali.

**Nuovi invarianti:**
1. nessuna sintesi della voce del soggetto;
2. ogni parola esportata ha provenienza e alternative;
3. adattatori, profili e lessico hanno versione e SHA-256 nel log;
4. revisione cieca come impostazione predefinita;
5. l'impronta vocale è mostrata come punteggio, mai come identificazione.

---

## 5. Architettura tecnica

### 5.1 Motore e interfaccia

- **Motore:** il pacchetto Python `akoe` esistente, esteso. Espone un'API locale per l'interfaccia (HTTP su localhost o stdio).
- **Interfaccia:** app desktop **Tauri** (frontend web, guscio Rust).
  - Sul portatile il motore gira come processo accanto all'app (sidecar).
  - Su workstation o cloud gira su quella macchina, e l'app vi si collega in rete cifrata.
  - L'interfaccia è la stessa su tutti i livelli.
- **Peso:** Python con PyTorch occupa alcuni GB; i modelli si scaricano a parte, non stanno nell'installer.
- **Perché Tauri e non Slint.** Kalliope (progetto sviluppato sulla stessa macchina, `/workspace/kalliope`) usa Slint. Per Akoe si sceglie Tauri perché forma d'onda, spettrogramma e revisione del testo sono più semplici con tecnologie web.

### 5.2 Livelli hardware

| Livello | Hardware | Cosa ci gira |
|---|---|---|
| EDGE | Portatile senza GPU (riferimento: Ryzen 5 PRO 6650U, 14 GB) | Interfaccia, pulizia riproducibile, impronta vocale, trascrizione con modelli piccoli o con adattatori già calcolati; adattamento solo di modelli piccoli, lento |
| WORKSTATION | GPU NVIDIA (consigliati almeno 12-16 GB) | Adattamento del trascrittore in ore; modelli grandi; pulizia neurale |
| CLOUD | VM con GPU sotto controllo dell'utente | Adattamenti pesanti e modelli più grandi; vincoli nella sezione 7.1 dell'`IMPLEMENTATION_PLAN.md` |

**Requisiti (3.4.3).** Non esistono cifre ufficiali e mancano tempi misurati senza GPU. Le esecuzioni documentate di LoRA su Whisper large usano tra 12 e 20 GB di memoria GPU. Sul portatile senza GPU l'adattamento è realistico solo per modelli piccoli (tiny/small), con tempi lunghi; i modelli grandi si adattano su workstation o cloud e l'adattatore si usa poi anche sul portatile.

**Vincolo invariato:** niente servizi SaaS di terzi; l'audio passa solo per macchine controllate.

### 5.2.1 Configurazioni di installazione

L'app parla con il motore solo attraverso l'API, quindi lo stesso Akoe si installa in quattro modi:

| Configurazione | Dove gira l'app | Dove gira il motore | Esempio |
|---|---|---|---|
| **Tutto in uno** | Sulla stessa macchina | Sulla stessa macchina, con GPU | Workstation Linux potente; NVIDIA DGX (Spark, Station o server) con la sua distribuzione Linux |
| **GPU in rete locale** | Portatile o postazione del perito | Su una DGX o un server GPU collegato in LAN, mai esposto a internet | La postazione dell'operatore con una DGX in laboratorio |
| **Solo portatile** | Portatile | Portatile (sidecar), modelli più piccoli | Lavoro in trasferta |
| **Cloud controllato** | Postazione | VM con GPU dell'utente (vincoli dell'`IMPLEMENTATION_PLAN.md` §8) | Nemotron 3.5 Transcribe su H100 |

Requisiti da coprire in M8:
- connessione cifrata e autenticata tra app e motore in rete locale (TLS o tunnel SSH, token);
- build CUDA del motore, anche per **ARM64**: la DGX Spark ha CPU ARM e memoria unificata da 128 GB, quindi pacchetti Python e app vanno provati su aarch64;
- distribuzione dei job tra più GPU;
- misura dei requisiti reali per ciascuna configurazione (#45).

Con molta memoria GPU (DGX) entrano anche i modelli più grandi del catalogo (Nemotron 3.5 Transcribe, Voxtral Small, omniASR 7B).

### 5.3 Lezioni da Kalliope

Da Kalliope si riprende il metodo, non il modello:
- **Inferenza in Rust con ONNX Runtime e pesi quantizzati**, per i componenti che devono girare bene su CPU.
- **Esportazione dei modelli verificata numericamente** contro il codice di riferimento, con soglie di tolleranza (come fa `tools/pocket_export/export.py`). Dimostra che il modello convertito produce gli stessi numeri dell'originale: un requisito di riproducibilità utile in perizia.

### 5.4 Modello dati

Estensioni previste a `models.py`:
- alternative per parola e per segmento, con punteggi;
- `SpeakerProfile`: impronta, profilo spettrale, dati di adattamento, quota di autovalutazione;
- `VehicleProfile`: risposte all'impulso, profili di rumore;
- `AdapterArtifact`: modello di base, dati, iperparametri, SHA-256;
- `SelfAssessment`;
- provenienza per parola nell'esportazione.

### 5.5 Catena di custodia

Si estende dai file audio a profili, adattatori, lessico e risultati dell'autovalutazione. Ogni artefatto ha SHA-256, parametri, versione del software e dei modelli.

---

## 6. Implementazione: stato attuale

Aggiornato il 2026-11-20. Repository privato `SANSEVIERIA-OU/akoe`; dettaglio per modulo in `IMPLEMENTATION_PLAN.md` (sezioni 3 e 4) e nelle issue.

**Motore completo, collegato end-to-end** (`engine.py`, raggiungibile da API e CLI):
- casi con reperti in sola lettura e hash, coppie locale/trasmesso, ruolo "riferimento" per il materiale del soggetto;
- modello del soggetto multi-contesto: import per condizione, allineamento CTC dei testi, divisione per registrazione, profilo timbrico, impronta vocale (ReDimNet-B2 raccomandato, ECAPA-TDNN di riferimento) con calibrazione leave-one-recording-out, ciclo di verifica;
- profilo del veicolo misurato (runbook e strumenti di misura) o stimato; simulazione del canale; codec variabili (G.711, GSM, MP3, Opus, Vorbis, AMR-NB, AAC; AMR-WB se c'è l'encoder); allineamento e stima del canale tra registrazione locale e trasmessa;
- pulizia: bande SII, riduzione del rumore offline, armoniche del motore, cancellazione di radio/musica con riferimento, pulizia d'ascolto NMF del soggetto, rimescolamento, rilevatore di voce robusto;
- trascrizione con alternative: Whisper (N-best vero, campionamento, decoder forzato, adattatori), Parakeet via NeMo (codice pronto, 0.6B v3 non ancora eseguito per limiti di RAM), combinazione stile ROVER, guardia contro le allucinazioni, lessico con letture con/senza; punteggio "X o Y?" con CTC e Whisper separati;
- adattamento: dati multi-contesto dai soli tratti verificati, LoRA o completo, adattatori con hash verificati al caricamento;
- autovalutazione per caso e per condizione con sintesi in italiano;
- API con token, job per ogni flusso, streaming audio etichettato.

**Verifiche:** 338 test rapidi e 9 lenti. Il test lento end-to-end usa un clip italiano reale in rumore d'auto sintetico, whisper-tiny, il modello CTC italiano ed ECAPA: frase trascritta correttamente, frase vera prima per entrambi i punteggi; un adattatore addestrato per 2 passi peggiora nettamente e l'autovalutazione lo rileva.

**Limiti attuali:** tutte le soglie sono tarate su segnali sintetici o pochi clip pubblici; la qualità reale va misurata sul banco di prova (#48). Mancano l'app Tauri (M7, percorso di design da avviare), la distribuzione per livelli (M8) e l'esecuzione di Parakeet-TDT 0.6B v3.

---

## 7. Valutazione

Due livelli: l'autovalutazione per caso (4.7) e la valutazione durante lo sviluppo.

### 7.1 Metriche

- **Trascrizione:**
  - WER e CER;
  - copertura delle alternative (WER oracolo sulle N migliori);
  - calibrazione della confidenza (diagrammi di affidabilità, ECE);
  - tasso di inserimento, in particolare di termini del lessico non pronunciati;
  - correttezza delle etichette "incomprensibile".
- **Impronta vocale:** tasso di errore a pari errore (EER) e calibrazione dei punteggi sul materiale del caso.
- **Pulizia:**
  - effetto sul trascrittore automatico (WER prima e dopo);
  - per l'ascolto umano, test di intelligibilità quando possibile;
  - PESQ e STOI solo come indicatori secondari (per SWGDE le misure oggettive non sempre funzionano in ambito forense).
- **Pipeline:** integrità della provenienza (100%) e completezza della catena di custodia.

### 7.2 Dati

- **Sintetici (oggi):** solo per i test di funzionamento.
- **Banco di prova del prodotto:** registrazioni in auto con microfono simile e testo noto, prodotte durante lo sviluppo. Servono a verificare i moduli; non sono un esperimento a sé.
- **Reali:** materiale di casi, usabile solo internamente.
- **Pubblici:** nessuno su registrazioni nascoste in auto. I benchmark in auto esistenti (AISHELL-5, ICMC-ASR) sono in cinese, con parlanti collaborativi.

---

## 8. Limitazioni note e sfide aperte

1. **Informazione assente nel segnale.** Nessun modello recupera ciò che il rumore ha coperto del tutto; può solo indovinare. Il prodotto deve astenersi.
2. **Soliloqui e sforzo vocale.** Il materiale di riferimento (telefonate, interrogatori) ha uno sforzo vocale diverso dal parlato a bassa voce o dal parlato alzato per il rumore in auto. La differenza di sforzo porta l'EER dei sistemi di riconoscimento del parlante oltre il 20% (3.5.3). Sul parlato autodiretto non esistono dati di trascrizione automatica. Mitigazioni: profili per condizione e uso delle registrazioni precedenti nella stessa auto.
3. **Dialetto e italianizzazione** (3.9).
4. **Materiale trascritto del soggetto** scarso o di qualità incerta: le trascrizioni esistenti possono essere a loro volta errate o italianizzate.
5. **Pulizia che peggiora il trascrittore automatico** (1.4, 3.4).
6. **Bias verso le parole attese** introdotto dal lessico del caso (1.4).
7. **Fattore umano:** priming e fiducia eccessiva nelle ipotesi della macchina (3.2).
8. **Chi guida.** Il soggetto potrebbe non essere alla guida, o esserci passeggeri occasionali: serve la verifica del parlante su ogni tratto.
9. **Nessun benchmark pubblico** su registrazioni nascoste in auto.
10. **Licenze.** Alcuni modelli candidati sono CC-BY-NC; i pesi di Pocket TTS hanno termini sul consenso; le licenze dei modelli di verifica del parlatore sono da verificare.
11. **Inquadramento legale:** considerato soddisfatto per decisione del proprietario (software per le forze dell'ordine, 2026-11-20); restano i termini di licenza dei modelli (#46).
12. **Ammissibilità.** La trascrizione resta del trascrittore; Akoe è uno strumento di supporto e la prova resta la registrazione (1.3).

---

## 9. Decisioni di progetto — log

| Data | Decisione | Motivazione | Stato |
|---|---|---|---|
| 2026-05-25 | Doppia modalità FORENSIC/INVESTIGATIVE come architettura primaria | Servono sia supporto investigativo interno sia materiale rigoroso per perizia | Implementato |
| 2026-05-25 | Punto di partenza: prototipo CLI Python, nessuna UI | Velocità di iterazione | **Superato** il 2026-11-20 (interfaccia Tauri) |
| 2026-05-25 | Materiale di riferimento del soggetto gestito in modo adattivo (poco → solo embedding; tanto → fine-tuning) | La quantità di campioni varia per caso | **Ripreso** il 2026-11-20 come adattamento del trascrittore |
| 2026-05-25 | La pipeline gestisce sia rumore dominante sia voci sovrapposte | Realismo | **Superato** il 2026-11-20: scenario primario con soggetto solo |
| 2026-05-25 | Hardware non vincolante nel prototipo | Priorità alla qualità | **Superato** il 2026-11-20: tre livelli espliciti |
| 2026-05-25 | Deployment non deciso, licenze permissive per il prototipo | Fase esplorativa | In essere |
| 2026-05-25 | Selezione tecnologica dipendente dalla modalità (forense conservativo, investigativo allo stato dell'arte) | Correzione di un bias (10.1) | In essere |
| 2026-05-25 | Includere esplicitamente ricerca cinese e israeliana | Correzione di un bias (10.1) | In essere |
| 2026-05-25 | Asse Tier (EDGE/WORKSTATION/CLOUD) ortogonale a Mode | Uso su hardware diversi | Confermato il 2026-11-20; refactor da fare |
| 2026-05-25 | CLOUD solo su VM controllate dall'utente, niente SaaS | Riservatezza dei dati di intercettazione | In essere |
| 2026-05-25 | Catena di custodia SHA-256 obbligatoria | Riproducibilità per perizia | Implementato; da estendere a profili e adattatori |
| 2026-05-25 | Audio deterministico e neurale mai mescolati | Invariante probatorio | Implementato |
| 2026-11-20 | Scenario primario: soggetto da solo in auto, anche in soliloquio | Frequente; elimina il rischio di attribuzione tra parlanti; canale misurabile | Deciso |
| 2026-11-20 | Obiettivo: prodotto sperimentale, non esperimento separato; validazione come autovalutazione integrata per caso | Scelta dell'utente; la misura resta, ma dentro il prodotto | Deciso |
| 2026-11-20 | Estrazione del parlante (TSE) accantonata per lo scenario primario | Benchmark sintetici, rischio di attribuzione, non necessaria con un solo parlante | Deciso |
| 2026-11-20 | Impronta vocale discriminativa (verifica del parlatore) | Serve a riconoscere chi parla, selezionare materiale, condizionare la pulizia; lo "stato voce" di Kalliope serve a sintetizzare, non a riconoscere | Deciso; modello da scegliere |
| 2026-11-20 | Profilo timbrico-spettrale del soggetto da registrazioni reali | Tarare la separazione voce/rumore sulla sua voce | Deciso; metodo dopo la ricerca (3.5) |
| 2026-11-20 | Esclusa la clonazione della voce del soggetto | Nessuna informazione in più sul timbro, termini d'uso sul consenso, rischio processuale, nessun dialetto | Deciso |
| 2026-11-20 | Profilo del veicolo misurato (risposta all'impulso, rumore) | Gli inquirenti hanno accesso all'abitacolo; riduce la distanza tra laboratorio e campo | Deciso |
| 2026-11-20 | Adattamento del trascrittore a soggetto e canale; "si addestra dove c'è potenza, si usa ovunque" | Evidenze sull'adattamento al parlante; adattatori leggeri e portabili | Deciso |
| 2026-11-20 | Risultato come interpretazioni multiple, confronto tra candidate, etichetta "incomprensibile" | Incertezza esplicita; l'analisi acustica serve soprattutto a escludere | Deciso; metodo dopo la ricerca (3.6) |
| 2026-11-20 | Revisione cieca come impostazione predefinita | Priming e trascrittori "ad hoc" (3.2) | Deciso |
| 2026-11-20 | Livelli hardware: tutti, dal portatile senza GPU al cloud | Richiesta dell'utente | Deciso |
| 2026-11-20 | Interfaccia: app desktop Tauri, motore Python locale o remoto | Una sola interfaccia per tutti i livelli; forma d'onda e spettrogramma più semplici con tecnologie web | Deciso |
| 2026-11-20 | Correzione bibliografica: il dato "~50%" su audio forense è di Loakes 2024, non di Fraser & Maher | Errore di attribuzione nella versione di maggio | Corretto |
| 2026-11-20 | La conoscenza del timbro va nel trascrittore automatico (adattamento), non in un filtro davanti; la pulizia personalizzata serve all'ascolto umano; al trascrittore va l'originale o il pulito rimescolato, scelto per caso | La pulizia davanti a Whisper di solito peggiora (3.4.1); i modelli specifici del parlante aiutano gli ascoltatori (3.5.1) | Deciso |
| 2026-11-20 | Profili del soggetto per condizione; materiale preferito: registrazioni precedenti nella stessa auto verificate dal trascrittore (ciclo di verifica) | Lo sforzo vocale e il canale cambiano la voce (3.5.3) | Deciso |
| 2026-11-20 | Trascrittori proposti: Parakeet-TDT 0.6B v3 (principale), Whisper large-v3 (seconda famiglia), modello CTC italiano (allineamento e candidate) | Confidenza entropica e WER basso; errori diversi tra famiglie; il punteggio delle candidate richiede CTC | Proposta, da verificare nel prodotto |
| 2026-11-20 | Alternative da fonti interpretabili più campionamento/MBR; confidenza entropica calibrata; guardia contro le allucinazioni; candidate in piccoli insiemi con soglie per caso | Evidenze in 3.6 | Deciso |
| 2026-11-20 | Adattamento: LoRA come impostazione predefinita, fine-tuning completo come opzione | Il completo è pari o di poco migliore; LoRA è portabile (3.4.3) | Deciso |
| 2026-11-20 | Modello del soggetto multi-contesto: impronta e adattatore addestrati su tutte le occasioni (telefono, stanze, auto) con simulazione di canali e codec; specializzazione per condizione solo se serve | Idea guida dell'utente; l'addestramento multi-condizione è la ricetta più robusta (3.4.2) | Deciso |
| 2026-11-20 | Microspia con registrazione locale e trasmissione: due reperti per evento; la locale è l'ingresso predefinito; la coppia serve a misurare il canale di trasmissione | Informazione dell'utente | Deciso |
| 2026-11-20 | Codec variabili: rilevati per file e simulati in addestramento (G.711, MP3, Opus, Vorbis; AMR/GSM se c'è un encoder) | Informazione dell'utente; robustezza al codec | Deciso |
| 2026-11-20 | Repository privato `SANSEVIERIA-OU/akoe` con roadmap (milestone M0-M8) e issue | Richiesta dell'utente | Deciso |
| 2026-11-20 | Verifica legale considerata soddisfatta (#47) | Decisione del proprietario: software per le forze dell'ordine | Deciso |
| 2026-11-20 | Progetto rinominato da WHISPER ad **Akoe**; pacchetto Python `akoe`; repository `SANSEVIERIA-OU/akoe` | Evitare la confusione con il modello Whisper di OpenAI | Deciso |
| 2026-11-20 | Licenza SANSEVIERIA License 1.1 (LICENSE, NOTICE, THIRD_PARTY_NOTICES, CONTRIBUTING, intestazione SPDX in ogni file) | Richiesta del proprietario, come gli altri progetti SANSEVIERIA | Deciso |
| 2026-11-20 | Criterio di scelta dei modelli: il migliore disponibile al mondo, con l'unico vincolo di girare su macchine controllate (portatile, workstation, cloud dell'utente) | Richiesta del proprietario | Deciso |
| 2026-11-20 | Pesi pubblicati senza licenza propria in un repository con licenza dichiarata: si considerano coperti dalla licenza del repository (es. ReDimNet, MIT) | Decisione del proprietario | Deciso |
| 2026-11-20 | Catalogo modelli per livello (`docs/catalogo_modelli.md`): trascrizione Cohere Transcribe + Qwen3-ASR + Parakeet su tutti i livelli, Nemotron 3.5 Transcribe in cloud, Whisper per l'adattamento; impronta ReDimNet2-B6 (w2v-BERT 2.0 per la massima accuratezza); CTC omniASR; pulizia d'ascolto UniPASE/GTCRN; Silero VAD | Ricerca del 2026-11-20 (3.11) e criterio "il migliore disponibile" | Deciso; backend da implementare |

---

## 10. Riflessioni metodologiche

Questa sezione documenta le correzioni metodologiche emerse durante il progetto. È materiale per la sezione metodologica del paper: mostra come i bias dell'analista, umano o AI, portino a scelte sub-ottimali se non vengono contrastati esplicitamente.

### 10.1 Bias nella selezione tecnologica e sua correzione (maggio 2026)

**Errore iniziale.** Nella prima proposta di stack era stato applicato uniformemente il criterio "preferisci ciò che è citato in peer review". L'utente ha obiettato che:
1. la peer review ha una latenza di 12-18 mesi, quindi privilegiare le citazioni esclude lo stato dell'arte reale degli ultimi 12 mesi;
2. l'addestramento dell'assistente era sovra-rappresentato sulla ricerca anglofona (Interspeech/ICASSP) e sotto-rappresentato su quella cinese e israeliana;
3. di conseguenza il software "nasce vecchio".

**Correzione.** Una regola dipendente dalla modalità: conservatorismo in modalità forense, stato dell'arte recente in modalità investigativa, indipendentemente dalla lingua di pubblicazione. È stata verificata con un'indagine esplicita affidata a tre agenti paralleli.

### 10.2 Assunzioni di contesto non verificate (maggio 2026)

**Errore iniziale.** Da "aiutare le forze dell'ordine" era stato dedotto "applicazione governativa", e la deduzione era stata usata come vincolo (filtro sulle licenze CC-BY-NC).

**Correzione.** La clausola NonCommercial si valuta caso per caso. Più in generale: non dedurre vincoli di ambito (governativo, commerciale, open source) senza verificarli con l'utente.

### 10.3 Implicazioni per il paper

Un assistente AI usato per lo scoping tecnico produce scelte sub-ottimali se non viene vincolato esplicitamente a contrastare i propri bias, linguistici e deduttivi. L'indagine con più agenti in parallelo è uno strumento riutilizzabile per farlo.

### 10.4 Scetticismo eccessivo e sua correzione (novembre 2026)

**Errore.** Nella revisione critica l'assistente ha affermato, a memoria, che "su parlato degradato l'orecchio umano batte quasi sempre la macchina". L'utente ne ha dubitato. La ricerca ha trovato evidenze contrarie in condizioni controllate (Patman & Chodroff 2024; Kim et al. 2024).

**Lezione.** Anche i giudizi critici vanno verificati sulle fonti: lo scetticismo dell'assistente può essere a sua volta un bias. La critica ha comunque prodotto cambiamenti utili (scenario, incertezza esplicita, revisione cieca).

### 10.5 Attribuzione bibliografica errata (corretta a novembre 2026)

La versione di maggio attribuiva a "Fraser & Maher (2024)" il dato "best ~50% correct on indistinct forensic audio" e citava il lavoro di Frontiers come "Speech analysis in forensic contexts". Il dato e l'articolo sono di Loakes (2024), "Automatic speech recognition and the transcription of indistinct forensic audio". L'articolo JAES di Fraser, Aubanel, Maher et al. (2024) si intitola "Forensic speech enhancement"; il suo contenuto non è stato letto.

### 10.6 Riferimenti ambigui (novembre 2026)

"Un modello simile a quello del progetto Kalliope" è stato interpretato come un progetto esterno, e l'assistente ha avviato una ricerca sul web. Era invece il progetto dell'utente sulla stessa macchina. **Lezione:** controllare il workspace prima del web quando un nome non è noto.

### 10.7 Esperimento o prodotto (novembre 2026)

L'assistente ha proposto un esperimento di fattibilità come primo passo; l'utente ha scelto di costruire un prodotto sperimentale. La sintesi: la misura non scompare, diventa l'autovalutazione per caso integrata nel prodotto (4.7).

---

## 11. Sviluppi futuri

- **Scenari con più parlanti** (auto con passeggeri, ambienti chiusi): rientrano diarizzazione ed estrazione del parlante, con verifica del soggetto e test sul caso "soggetto assente".
- **Più microfoni**, se disponibili nell'installazione.
- **Apprendimento dalle correzioni del trascrittore**, per migliorare adattamento e calibrazione.
- **Integrazione con strumenti forensi esistenti** (acquisizione con catena di custodia già avviata).
- **Verifica di integrità e manomissione** della traccia prima dei trattamenti (ENFSI FSA-BPM-002).
- **Corpus dialettali** costruiti dal materiale dei casi, dove consentito.
- **Valutazione legale e procedurale** con periti e avvocati prima dell'uso operativo.

---

## 12. Riferimenti

### 12.1 Macchina e orecchio umano; trascrizione forense

- Patman, C. & Chodroff, E. (2024). "Speech recognition in adverse conditions by humans and machines." JASA Express Letters 4:115204. doi:10.1121/10.0032473
- Kim et al. (2024). "Automatic recognition of second language speech-in-noise." JASA Express Letters 4:025204. doi:10.1121/10.0024877
- Loakes, D. (2024). "Automatic speech recognition and the transcription of indistinct forensic audio." Frontiers in Communication 9. https://www.frontiersin.org/articles/10.3389/fcomm.2024.1281407/full
- Loakes, D. (2022). Frontiers in Communication 7:803452 (allineamento forzato su audio forense scadente)†
- Spille, C., Kollmeier, B. & Meyer, B. (2018). Computer Speech & Language 52:123-140. doi:10.1016/j.csl.2018.04.003 † (non verificato)
- Lippmann, R. (1997). "Speech recognition by machines and humans." Speech Communication 22(1):1-15 †
- Koenecke, A. et al. (2024). "Careless Whisper: Speech-to-Text Hallucination Harms." FAccT '24. arXiv:2402.08021 †
- Love, R. & Wright, D. (2021). Frontiers in Communication 6:797448 †
- Fraser, H. (2018). "Assisting listeners to hear words that aren't there." Australian Journal of Forensic Sciences. doi:10.1080/00450618.2017.1340522 † (dati da fonte secondaria)
- Fraser, H. & Kinoshita, Y. (2021). "Injustice arising from the unnoticed power of priming." Criminal Law Journal 45(3):142-152 †
- Fraser, H., Aubanel, V., Maher, R. et al. (2024). "Forensic speech enhancement." JAES 72(11):748-753. doi:10.17743/jaes.2022.0176 † (solo metadati)
- French, P. & Fraser, H. (2018). "Why 'ad hoc experts' should not provide transcripts of indistinct forensic audio." Criminal Law Journal 42:298-302. https://eprints.whiterose.ac.uk/id/eprint/170253/
- Fraser, H. (2021). "Forensic Transcription: Legal and scientific perspectives." Studi AISV 8. doi:10.17469/O2108AISV000001
- Simonotti et al. (2026). arXiv:2603.16258 (trascrizione assistita da ASR, italiano, KIParla)
- Kuhn, Kersken & Zimmermann (2025). CHI EA '25. arXiv:2503.15124
- Stoykova, Porter & Beka (2024). Forensic Science International: Synergy 9:100563 †
- Harrington, L. (2024). Tesi di dottorato, University of York. https://etheses.whiterose.ac.uk/id/eprint/35134/ †

### 12.2 Acustica, pulizia, linee guida, trascrizione in auto

- ANSI/ASA S3.5-1997, Speech Intelligibility Index (pesi d'ottava dal pacchetto R `SII`)
- Alexander, A., Forth, O. & Tunstall, D. (2012). Fingerprinting e cancellazione con riferimento. AES 46th Conference. https://oxfordwaveresearch.com/papers/AES-2012-FingerprintingCancellationAlexanderForthTunstall.pdf
- Erkek & Ozcan (2026). arXiv:2606.18564 †
- SWGDE 20-A-001-2.0 (2025). Best Practices for the Enhancement of Digital Audio. https://swgde.org/wp-content/uploads/2025/09/2025-08-21-Best-Practices-for-Enhancement-of-Digital-Audio-20-A-001-2.0.pdf
- ENFSI FSA-BPM-002 (2022). Best Practice Manual for Digital Audio Authenticity Analysis. https://enfsi.eu/wp-content/uploads/2022/12/FSA-BPM-002_BPM-for-Digital-Audio-Authenticity-Analysis.pdf
- ICMC-ASR (ICASSP 2024). arXiv:2401.03473 †
- AISHELL-5 (Interspeech 2025). https://www.isca-archive.org/interspeech_2025/dai25c_interspeech.pdf
- Hansen et al. (2000, 2001). CU-Move. ICSLP 2000; Eurospeech 2001
- Lee et al. (2004). AVICAR. ICSLP 2004
- Weng et al. (2016). Conversational in-vehicle dialog systems. IEEE Signal Processing Magazine

### 12.3 Adattamento al parlante e al canale

- Shor, J. et al. (2019). arXiv:1907.13511 †
- Tobin, J. & Tomanek, K. (2021). arXiv:2110.04612 †
- Green, J. et al. (2021). Interspeech 2021 †
- Huber & Waibel (2026a). arXiv:2606.31722 (adattamento di Whisper a un singolo parlante)
- Huber & Waibel (2026b). arXiv:2608.05759 (sovra-biasing contestuale e inserimenti)
- van Dalen et al. (2025). arXiv:2506.10653
- Mihajlik et al. (2025). arXiv:2508.06391
- Senthil & Fokoué (2026). arXiv:2607.27245 (body-cam delle forze dell'ordine)
- Eskimez, S. et al. (2021). Personalized speech enhancement. arXiv:2110.09625
- arXiv:2212.00406 (pulizia personalizzata che peggiora l'accuratezza)
- IR-GAN (2020). arXiv:2010.13219
- CabinSep (2025). arXiv:2509.01399
- arXiv:2205.01751 (MixIT, controprova su CHiME-3)
- Liu et al. (2025). arXiv:2508.17796 (biasing contestuale per Whisper)
- ROXANNE D5.3 (2023), Horizon 2020 GA 833635. https://www.roxanne-euproject.org/results/files/d5-3_final_set_of_roxanne_speech_nlp_video_technologies_for_network_analysis_v1-0.pdf
- Cacioli (2026). MRL 2026. https://aclanthology.org/2026.mrl-1.42/ †

### 12.4 Fonetica forense e attribuzione

- Cole, R., Rudnicky, A., Reddy, R. & Zue, V. (1978). JASA 64:S178. doi:10.1121/1.2004042 †
- Zue, V. & Cole, R. (1979). JASA 65:S81 †
- Morrison, G. (2009). "Forensic voice comparison and the paradigm shift." Science & Justice. doi:10.1016/j.scijus.2009.09.002
- Morrison, G., Lindh, J. & Curran, J. (2014). Speech Communication 58:81-90 †
- Drygajlo, A. et al. (2015). ENFSI Methodological Guidelines for Semiautomatic and Automatic Speaker Recognition. https://enfsi.eu/wp-content/uploads/2016/09/guidelines_fasr_and_fsasr_0.pdf
- Cenceschi, S. & Meluzzi, C. (2023). Studi AISV 10. https://www.aisv.it/StudiAISV/2023/vol_10/studiAISV_10.pdf
- Paoloni, A. Atti AISV (anno da confermare). https://www.aisv.it/PubblicazioniAISV/III_AISV/papers/pdf/METODOLOGIE%20DI%20VALUTAZIONE/Paoloni.pdf

### 12.5 Normativa e prassi italiana

- Codice di Procedura Penale, artt. 266-271; art. 268 c. 7
- D.Lgs. 216/2017; D.L. 161/2019, conv. L. 7/2020
- Corte di Cassazione, Ufficio del Massimario, Relazione n. 35/2020. https://www.cortedicassazione.it/resources/cms/documents/Rel_3520.pdf
- AISV-OLF (2019). Proposta di linee guida per la perizia fonica. https://olf.aisv.it/2019/03/01/proposta-di-linee-guida-per-lo-svolgimento-di-una-perizia-fonica-scientificamente-accettabile/ †
- Meluzzi, C. (2025). Forensic transcription and the challenges of segmental and suprasegmental cues (preprint, Università di Milano)
- Romito (2025). Camera penale di Catanzaro †
- Di Nunno (2025). Abstract IAFPA 2025, p. 20
- Procura generale di Perugia, linee guida sull'AI (Umbria24, 7/9/2026) † (testo non letto)

### 12.6 Progetti e strumenti

- Kalliope (progetto sulla stessa macchina): `/workspace/kalliope`
- Pocket TTS (Kyutai): https://huggingface.co/kyutai/pocket-tts
- Tauri: https://tauri.app
- Riferimenti del censimento di maggio 2026: Appendice D

### 12.7 Pulizia informata dal parlante, timbro, sforzo vocale

- Mohammadiha, N., Smaragdis, P. & Leijon, A. (2013). "Supervised and unsupervised speech enhancement using NMF." IEEE TASLP. arXiv:1709.05362
- Mohammadiha, N. (2013). Tesi di dottorato, KTH. diva-portal.org diva2:637931
- Srinivasan, S. (2005). Tesi di dottorato, KTH (codebook). diva-portal.org diva2:12853
- Chidambar & Hanumantha Rao Naidu (2024). JASA Express Letters 4(11):115203 (codebook adattato al parlante)
- Ouyang et al. (2018). Interspeech 2018 (modello armonico con DNN)
- Leer et al. (2026). arXiv:2607.04826 (modelli specializzati sul parlante; preprint)
- Sivaraman, A. & Kim, M. (2022). IEEE JSTSP. arXiv:2104.02017
- Kolbæk, M., Tan, Z.-H. & Jensen, J. (2017). IEEE/ACM TASLP (DNN generali e specializzate) †
- Goehring, T. et al. (2017). Hearing Research. https://pmc.ncbi.nlm.nih.gov/articles/PMC5256482
- Arantes, P., Eriksson, A. & Gutzeit, S. (2017). Interspeech 2017 (stabilizzazione di F0) †
- Heald, S. & Nusbaum, H. (2015). PLoS One 10(9):e0136791
- Shadle, C., Nam, H. & Whalen, D. (2016). JASA 139(2):713
- Sukhdeve et al. (2026). arXiv:2611.13258
- Byrne, D., Dillon, H. & Tran, K. (1994). JASA 96(4):2108 †
- Gołębiowska & Syga (2026). arXiv:2604.20229 (verifica del parlante e sussurrato)
- López-Espejo et al. (2023). arXiv:2305.02147 (sforzo vocale)
- Lee et al. (2025). arXiv:2511.07493 (MutterMeter, parlato autodiretto) †
- Simonotti et al. (2026). "Say Again?" ACL 2026 speakable-1.3 †

### 12.8 Pulizia e trascrittore automatico, simulazione, adattamento

- Chondhekar et al. (2025). "When De-noising Hurts." arXiv:2512.17562
- Huo et al. (2026). "Where Speech Enhancement Hurts Recognition." arXiv:2607.11157
- Wang et al. (2024). "Bridging the Gap." arXiv:2406.12699
- Cui et al. (2025). arXiv:2501.02452
- Sato et al. (2025). Frontiers in Signal Processing (NTT)
- Wang (2025). SuperM2M. arXiv:2403.10271
- Yang et al. (2023). FAT-HuBERT. arXiv:2311.17790
- de Groot et al. (2025). Interspeech 2025. arXiv:2508.17980
- Pranida et al. (2025). arXiv:2509.25878
- Boonpramuk et al. (2026). arXiv:2608.24916
- Oglic et al. (2022). arXiv:2110.08634
- Szöke et al. (2019). BUT ReverbDB. arXiv:1811.06795
- Tang et al. arXiv:1910.10815
- Balam et al. (2020). arXiv:2010.12715
- Fan et al. (2024). arXiv:2406.10507
- arXiv:2606.07608 (svizzero tedesco, LoRA contro completo)
- arXiv:2605.08186 e arXiv:2603.05231 (adattamento al momento della trascrizione)
- Lin et al. (2024). EMNLP 2024. arXiv:2406.11064
- Open ASR Leaderboard. arXiv:2510.06961
- NVIDIA Parakeet-TDT 0.6B v3 e Canary-1B-v2 (schede Hugging Face); OpenAI Whisper (github.com/openai/whisper)

### 12.9 Alternative, confidenza, allucinazioni, candidate

- Chen et al. (2023). HyPoradise. NeurIPS 2023. arXiv:2309.15701
- Jinnai (2025). arXiv:2510.19471 (MBR per la trascrizione)
- Novosad (2026). arXiv:2606.23306 †
- Fiscus, J. (1997). ROVER. NIST
- Parikh et al. (2024). LREC-COLING 2024 †
- Laptev, A. & Ginsburg, B. (2022). arXiv:2212.08703 †; documentazione NeMo sulla confidenza
- Huo et al. (2025). ASRU 2025. arXiv:2509.07195
- Aggarwal et al. (2025). C-Whisper. ICASSP 2025. arXiv:2502.13446
- Barański et al. (2025). ICASSP 2025. arXiv:2501.11378
- Barański et al. (2026). HALAS. arXiv:2606.23048 †
- Aparin et al. (2026). arXiv:2606.07473 †
- Atwany et al. (2025). Findings ACL 2025. arXiv:2502.12414
- Kürzinger et al. (2020). CTC-segmentation. arXiv:2007.09127 †
- NTC-KWS (2025). ICASSP 2025. arXiv:2412.12614
- Sun et al. (2023). HTEC. arXiv:2309.10089
- torchaudio `forced_align` (documentazione); NeMo Forced Aligner (documentazione)

---

## Appendice A — Cronologia del progetto

### A.1 Prima sessione (2026-05-25)

1. Idea iniziale: applicazione per pulire l'audio delle intercettazioni e comprendere il parlato del soggetto, sfruttando AI e campioni della sua voce in condizioni normali.
2. Scoping con domande strutturate: doppia modalità, materiale variabile, hardware non vincolante, scenario audio misto.
3. Scaffolding del progetto Python (stadi modulari, CLI, modelli Pydantic, smoke test).
4. Implementazione degli stadi deterministici e validazione della modalità forense su audio sintetico.
5. Critica dell'utente sul bias verso peer review e ricerca anglofona; indagine con tre agenti paralleli; stack rivisto.
6. Critica dell'utente su un'assunzione implicita (deployment governativo) e su un'affermazione imprecisa (CC-BY-NC); correzione.
7. Modello a tre livelli hardware; scrittura di `IMPLEMENTATION_PLAN.md` e `MASTER_PLAN.md`.

### A.2 Seconda sessione (2026-11-20)

1. Revisione critica dell'impostazione: limiti della TSE sul campo, rischio di attribuzione, ipotesi "sotto soglia umana" messa in discussione.
2. L'utente restringe lo scenario (soggetto solo in auto) e contesta lo scetticismo sull'orecchio umano. Quattro ricerche parallele: umani contro macchine, acustica dell'auto, adattamento al parlante, fonetica forense. Correzione (10.4).
3. L'utente sceglie un prodotto sperimentale invece di un esperimento; la validazione diventa autovalutazione integrata.
4. L'utente chiede tutti i livelli hardware, propone Tauri e introduce l'impronta vocale. Chiarito che "Kalliope" è il suo progetto locale; distinzione tra stato voce per la sintesi e impronta per il riconoscimento.
5. Discussione sulla clonazione per studiare il timbro: obiettivo condiviso, strumento escluso; introdotto il profilo timbrico-spettrale da registrazioni reali.
6. Aggiornamento dei piani; quattro nuove ricerche sull'analisi del timbro e sulle interpretazioni multiple, integrate nelle sezioni 3.4-3.6. Conclusione principale: la conoscenza del timbro rende di più dentro il trascrittore (adattamento) che in un filtro davanti; la pulizia personalizzata serve all'ascolto umano.

### A.3 Memoria persistente

I file di memoria della prima sessione sono conservati localmente da Claude Code sulla macchina di sviluppo: `project_audio_forensics.md`, `feedback_audio_forensics_hallucination.md`, `feedback_tech_selection_bias.md`, `feedback_no_context_assumptions.md`, `reference_audio_forensics_sota_2025.md`. Riflettono l'impostazione di maggio 2026.

---

## Appendice B — Comando di esempio (stato attuale)

```bash
cd /workspace/WHISPER
python -m venv .venv && source .venv/bin/activate
pip install -e .
python scripts/synth_test_audio.py
python -m akoe process \
    --evidence data/evidence.wav \
    --speaker-samples data/speaker_samples \
    --output out/case_001 \
    --mode forensic \
    --case-id case_001
```

Output atteso in `out/case_001/`: `preprocessed.wav`, `enhanced_deterministic.wav`, `processing_log.json` con catena SHA-256.

Test: `pip install pytest && python -m pytest tests/ -v` (2 test verdi).

---

## Appendice C — Glossario

- **ASR**: riconoscimento automatico del parlato (trascrittore automatico).
- **WER / CER**: percentuale di parole / caratteri sbagliati rispetto a una trascrizione di riferimento.
- **WER oracolo (N-best)**: errore che si avrebbe scegliendo sempre, tra le N ipotesi proposte, la più vicina al vero; misura la copertura delle alternative.
- **N-best, reticolo, confusion network**: modi di rappresentare più ipotesi di trascrizione con i relativi punteggi.
- **Calibrazione**: corrispondenza tra confidenza dichiarata e accuratezza reale (es. le parole date al 90% sono giuste 9 volte su 10).
- **Allineamento forzato**: allineare un testo dato all'audio con un modello acustico; produce tempi e un punteggio di compatibilità.
- **TSE**: estrazione del parlante bersaglio da una miscela di voci.
- **VAD**: rilevamento dei tratti con voce.
- **SNR**: rapporto segnale/rumore, in dB (valori negativi: rumore più forte della voce).
- **SII**: Speech Intelligibility Index (ANSI S3.5), pesi delle bande di frequenza per l'intelligibilità.
- **Risposta all'impulso**: "impronta" acustica del percorso tra una posizione (es. bocca del guidatore) e il microfono; permette di simulare come suonerebbe lì qualsiasi registrazione.
- **Spettro medio di lungo periodo (LTAS)**: distribuzione media dell'energia della voce di una persona sulle frequenze.
- **F0**: frequenza fondamentale, cioè quanto è grave o acuta la voce.
- **Formanti**: risonanze del tratto vocale che caratterizzano vocali e timbro.
- **Effetto Lombard**: modifica involontaria della voce (più forte, più acuta, spettro diverso) quando si parla nel rumore.
- **Impronta vocale (speaker embedding)**: vettore che rappresenta l'identità vocale; qui prodotto da un modello di verifica del parlatore (es. ECAPA-TDNN).
- **EER**: tasso di errore a cui falsi rifiuti e false accettazioni si equivalgono.
- **Rapporto di verosimiglianza (LR)**: quanto un'osservazione è più probabile sotto un'ipotesi che sotto l'alternativa.
- **Smascheramento sequenziale**: rivelare le informazioni di contesto all'esaminatore solo dopo il suo primo giudizio indipendente.
- **LoRA / adattatore**: piccolo insieme di parametri aggiunto a un modello per specializzarlo senza modificarlo tutto.
- **PESQ / STOI / SI-SDR / DNSMOS**: misure oggettive di qualità, intelligibilità e distorsione.
- **RTF**: fattore di tempo reale (tempo di calcolo / durata dell'audio).
- **Catena di custodia**: tracciatura ininterrotta dei trattamenti subiti da un reperto digitale.
- **Provenienza**: origine dichiarata di un dato (`deterministic` o `neural`).

---

## Appendice D — Censimento dei modelli (2026-05-25)

Conservato dalla versione di maggio. I numeri non sono stati riverificati a novembre 2026. La TSE è fuori dallo scenario primario (sezione 2).

### D.1 Pulizia del parlato (denoising)

| Modello | Origine | Anno | Miglior benchmark (dataset) | Licenza | Repo / disponibilità |
|---|---|---|---|---|---|
| MossFormer2_SE_48K (ClearerVoice-Studio) | Alibaba Speech Lab (CN) | 2024 | PESQ 3.57, STOI 0.98 (DNS-2020) | Apache-2.0 | github.com/modelscope/ClearerVoice-Studio |
| DeepFilterNet 3 | FAU Erlangen-Nürnberg (DE) | 2023 | PESQ 3.17, STOI 0.944 (VoiceBank+DEMAND) | MIT/Apache-2.0 | github.com/Rikorose/DeepFilterNet |
| SGMSE+ / StoRM | Univ. Hamburg (DE) | 2022→2025 | PESQ/ESTOI su VB-DEMAND, EARS-WHAM 48 kHz | MIT | github.com/sp-uhh/sgmse |
| Resemble-Enhance | Resemble AI | 2023 | qualitativo 44.1 kHz | MIT | github.com/resemble-ai/resemble-enhance |
| AnyEnhance | CUHK-Shenzhen + Huawei (CN) | 2025 | unifica denoise/dereverb/declip/SR/TSE | da verificare | github.com/viewfinder-annn/AnyEnhance-v1 |
| GenSE | Tencent + NTU | 2025 | basato su LM, SOTA DNS-2020 | solo paper | arXiv:2502.02942 |
| USEMamba | Internazionale | 2025 | 2° URGENT 2025 Track 1 | da verificare | arXiv:2505.21198 |
| SNR-Progressive + Harmonic Comp. | area NPU (CN) | 2024 | 1° ICASSP 2024 MISP | solo paper | arXiv:2406.16317 |
| Estensione DFN (Technion) | Technion + Insoundz (IL) | 2025 | real-time + dereverb | codice da localizzare | mdpi.com/1424-8220/25/3/630 |

Nota (maggio 2026): MossFormer2 primario su WORKSTATION, DeepFilterNet 3 su EDGE, SGMSE+ per casi limite su CLOUD. Nota (novembre 2026): sono modelli generici, non informati dal parlante; i modelli generativi (SGMSE+, AnyEnhance, GenSE) possono inventare contenuto e vanno trattati come ipotesi.

### D.2 Estrazione del parlante bersaglio (TSE)

| Modello | Origine | Anno | Benchmark | Licenza | Repo |
|---|---|---|---|---|---|
| WeSep + Multi-Level Speaker Repr. | NEU+CUHK-SZ+Tencent+NUS | 2024 | 17.99 dB SI-SDRi Libri2Mix-360 | Apache-style (da verificare) | github.com/wenet-e2e/wesep |
| WeSep base (BSRNN) | NPU+Alibaba+CUHK-SZ+SJTU | 2024 | 16.57 dB Libri2Mix-360 | Apache-style | github.com/wenet-e2e/wesep |
| MossFormer2 TSE | Alibaba Tongyi Lab | 2024 | SOTA WSJ0-2/3mix, Libri2Mix, WHAM/WHAMR | Apache-2.0 | huggingface.co/alibabasglab/AV_MossFormer2_TSE_16K |
| X-TF-GridNet | IoA CAS | 2024 | SOTA Libri2Mix | da verificare | github.com/HaoFengyuan/X-TF-GridNet |
| USEF-TSE | Univ. Surrey + SJTU | 2024 | SOTA WSJ0-2mix, WHAM, WHAMR | CC-BY-NC 4.0 | github.com/ZBang/USEF-TSE |
| SoloSpeech | multi-org | 2025 | SOTA intelligibilità Libri2Mix | CC-BY-NC 4.0 | github.com/WangHelin1997/SoloSpeech |
| Binaural TSE con HRTF | Bar-Ilan, Gannot Lab | 2025 | SI-SDRi, PESQ, STOI | CC-BY-NC-ND 4.0 | arXiv:2507.19369 |
| SpEx+ | NUS + JD AI | 2020 | 12-13 dB WSJ0-2mix | GPLv3 | github.com/xuchenglin28/speaker_extraction_SpEx |
| SpeakerBeam | NTT + BUT | 2019-2020 | 10-12 dB | non commerciale | github.com/BUTSpeechFIT/speakerbeam |

Limiti (novembre 2026): benchmark su miscele sintetiche a due voci; confusione del bersaglio; estrazione anche con soggetto assente.

### D.3 Trascrizione automatica per l'italiano

| Modello | Origine | Anno | WER italiano | Confidenza per parola | Licenza | Costo locale |
|---|---|---|---|---|---|---|
| NVIDIA Parakeet-TDT 0.6B v3 | NVIDIA + FBK | 2025 | 3.00% FLEURS-IT; 5.58% Open ASR multi-IT | timestamp, confidenza non nativa | CC-BY-4.0 | ~2 GB RAM, CPU possibile |
| Microsoft Phi-4-Multimodal | Microsoft | 2025 | 4.80% Open ASR multi-IT | indiretta | MIT | ~12 GB VRAM |
| Whisper large-v3 | OpenAI | 2023 | 5.14% Open ASR multilingual | sì (faster-whisper) | Apache-2.0 / MIT | ~10 GB VRAM fp16 |
| NVIDIA Canary-1B v2 | NVIDIA + FBK | 2025 | 5.66% Open ASR multi-IT | parziale | CC-BY-4.0 | 6 GB |
| Mistral Voxtral-Mini-3B | Mistral | 2025 | 5.88% Open ASR multi-IT | indiretta | Apache-2.0 | ~9.5 GB VRAM |
| Whisper large-v3-turbo | OpenAI | 2024 | n.d. | sì | MIT | ~6 GB VRAM |
| whisper-large-v3-distil-it-v0.2 | community | 2025 | n.d. | sì | MIT | ~3 GB VRAM |
| SeamlessM4T-v2-large | Meta | 2023 | n.d. per IT | no | CC-BY-NC 4.0 | ~12 GB VRAM |
| FunASR / Paraformer | Alibaba | — | italiano non supportato | — | MIT | — |
| FireRedASR | Xiaohongshu | 2025 | italiano non supportato | — | Apache-2.0 | — |

Nota (novembre 2026): i WER sono su dati puliti (FLEURS, Open ASR) e non predicono le prestazioni su registrazioni nascoste in auto. La scelta va rifatta in base ad adattabilità, alternative e confidenza.

### D.4 Riferimenti del censimento

- Schröter et al. (2023). DeepFilterNet. arXiv:2305.08227
- Welker et al. (2022). SGMSE+. arXiv:2208.05830
- Zhao et al. (2024). MossFormer2. arXiv:2312.11825
- Wang et al. (2024). WeSep. arXiv:2409.15799
- Liu et al. (2024). Multi-Level Speaker Representation. arXiv:2410.16059
- Zeng et al. (2024). USEF-TSE. arXiv:2409.02615
- Wang et al. (2025). SoloSpeech. arXiv:2505.19314
- Hao et al. (2024). X-TF-GridNet. Information Fusion
- AnyEnhance arXiv:2501.15417; GenSE arXiv:2502.02942; USEMamba arXiv:2505.21198; SNR-Progressive arXiv:2406.16317
- Ellinson & Gannot (2025). arXiv:2507.19369
- Radford et al. (2022). Whisper; Bain et al. (2023). WhisperX. arXiv:2303.00747
- NVIDIA Parakeet-TDT 0.6B v3: https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3 ; Canary-1B v2: https://huggingface.co/nvidia/canary-1b-v2 ; Granary: https://huggingface.co/datasets/nvidia/Granary
- Open ASR Leaderboard. arXiv:2510.06961
- Progetti: ClearerVoice-Studio, DeepFilterNet, SGMSE, WeSep, WeSpeaker, USEF-TSE, SoloSpeech, X-TF-GridNet, Resemble-Enhance, Demucs, SpEx+, SpeakerBeam, faster-whisper, WhisperX, NeMo, noisereduce, pyannote.audio, librosa, SpeechBrain
