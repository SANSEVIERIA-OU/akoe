# Runbook — Misura acustica del veicolo

Procedura operativa per costruire il **profilo misurato del veicolo** (`MASTER_PLAN.md` 4.3, `IMPLEMENTATION_PLAN.md` M3):

1. la **risposta all'impulso** dell'abitacolo dalla posizione della testa del guidatore al microfono nascosto;
2. le **registrazioni del rumore** per condizione di marcia.

Le misure servono a simulare il canale per l'adattamento del trascrittore e per l'autovalutazione, e a stimare il rapporto segnale/rumore. Non modificano in alcun modo i reperti.

Strumenti software: `python -m forensicaudio.vehicle.measure` (generazione dello sweep e deconvoluzione) e `forensicaudio.vehicle.profile` (import nel caso).

---

## 1. Avvertenze

**Autorizzazioni.**
- La misura si fa solo nell'ambito di un accesso al veicolo autorizzato dall'autorità giudiziaria: durante l'installazione della microspia o in un accesso successivo.
- Prima dell'accesso verificare che il provvedimento copra anche le operazioni tecniche di misura.

**Sicurezza.**
- Chi guida **non** manovra apparecchiature. Per le misure in marcia servono almeno due persone: un autista e un operatore.
- Rispettare il codice della strada. Le velocità indicate sono indicative: non superare mai i limiti. La condizione "finestrini aperti" in autostrada si registra solo se le condizioni lo consentono.
- Lo sweep è un fischio crescente, ben udibile anche fuori dall'abitacolo. Misurare in un luogo riservato e chiuso (garage, area recintata), per non attirare l'attenzione.
- Volume: non superare il necessario (sezione 5). Usare cuffie chiuse per l'ascolto di controllo.

**Nessuna traccia.**
- Fotografare l'abitacolo **prima** di toccare qualsiasi cosa: sedili, specchietti, finestrini, bocchette, autoradio, oggetti presenti.
- A fine misura ripristinare tutto com'era e fotografare di nuovo.
- Controllare la batteria del veicolo se si usa il quadro acceso a motore spento.

**Riservatezza.** Le registrazioni restano su supporti e macchine sotto il controllo dell'ufficio. Nessun servizio esterno o cloud di terzi.

---

## 2. Materiale

| Voce | Note |
|---|---|
| Microspia installata, oppure un esemplare dello **stesso modello** | Se si usa un altro esemplare, va montato nella stessa posizione e con lo stesso fissaggio e rivestimento |
| Stazione di ricezione | Per registrare anche il percorso trasmesso, quando possibile |
| Altoparlante piccolo a gamma estesa | Diffusore attivo compatto (3-4 pollici), risposta il più possibile piatta almeno da 100 Hz a 10 kHz |
| Lettore | Computer portatile con uscita a 48 kHz. Disattivare equalizzatori, "miglioramenti audio", normalizzazione del volume ed effetti di sistema |
| Supporto per l'altoparlante | Asta o treppiede con morsetto, per tenerlo fermo alla posizione della testa del guidatore |
| Metro, livella, fotocamera | Per posizioni e documentazione |
| Fonometro (facoltativo) | Per annotare il livello di riproduzione |
| Cuffie chiuse | Ascolto di controllo |
| Due supporti di archiviazione | Copia master e copia di lavoro |
| Modulo di verbale | Sezione 9 |

---

## 3. Preparazione del segnale di misura

Generare lo sweep esponenziale (metodo di Farina) sul computer di lavoro:

```bash
python -m forensicaudio.vehicle.measure genera --out sweep/
```

Valori predefiniti:
- 48 kHz;
- sweep da 50 Hz a 20 kHz, durata 10 s;
- 3 ripetizioni, ciascuna seguita da 3 s di silenzio;
- 1 s di silenzio iniziale.

Il comando scrive:
- il file da riprodurre, per esempio `sweep_48000Hz_50-20000Hz_10s_x3.wav`;
- `sweep_params.json` con i parametri, lo SHA-256 del file e le versioni del software.

Annotare lo SHA-256 nel verbale. Il file va riprodotto **così com'è**: niente conversioni, tagli o normalizzazioni.

Se la microspia applica un controllo automatico del guadagno (AGC) o una riduzione del rumore non disattivabili, la misura ne risente. In quel caso:
- annotarlo;
- se possibile, misurare con un esemplare dello stesso modello con AGC e riduzione del rumore disattivati, e annotarlo.

---

## 4. Posizionamento

**Veicolo.**
- Fermo, in luogo silenzioso.
- Motore spento, finestrini chiusi, porte chiuse durante la misura.
- Ventilazione, autoradio e avvisi acustici spenti.
- Sedile del guidatore nella posizione abituale, se nota; altrimenti annotare quella usata.

**Altoparlante.** Simula la bocca del guidatore.
- Centro del cono all'altezza della bocca di un guidatore seduto: indicativamente 65-70 cm sopra il piano della seduta, 10-15 cm davanti al poggiatesta.
- Rivolto in avanti, come chi guarda la strada.
- Misurare e annotare: altezza dalla seduta, distanza dal poggiatesta, distanza in linea retta dal microfono.
- Fotografare da tre angolazioni.

**Microfono.**
- La microspia resta nella posizione di installazione definitiva: stesso orientamento, stesse coperture e rivestimenti.
- Se si usa un esemplare dello stesso modello, va nella stessa posizione con lo stesso montaggio.

**Persone.** Durante lo sweep nessuno in abitacolo. Se è inevitabile, una persona immobile sul sedile posteriore: annotarlo.

**Passeggero (facoltativo).** Ripetere la misura con l'altoparlante alla testa del passeggero anteriore.

---

## 5. Livello di riproduzione

1. Avviare la registrazione sul dispositivo e riprodurre la prima parte dello sweep.
2. Regolare il volume dell'altoparlante: lo sweep deve superare nettamente il rumore di fondo senza saturare la registrazione locale. Picco indicativo sotto -6 dBFS, senza tratti "squadrati".
3. Riferimento indicativo: circa 80 dB(A) alla posizione del microfono. Annotare il livello se c'è un fonometro.
4. Una volta fissato, il volume non si cambia più per tutta la sessione.

---

## 6. Registrazione della risposta all'impulso

1. Avviare la registrazione **locale** sul dispositivo e, se possibile, la ricezione della **trasmissione** nella stazione.
2. Annotare l'ora da un orologio di riferimento.
3. Pronunciare un annuncio di identificazione, per esempio: "Veicolo V01, sweep posizione guidatore, sessione A, ore 10:42". Poi silenzio.
4. Riprodurre il file di sweep: contiene le 3 ripetizioni. Durante la riproduzione silenzio assoluto e nessun movimento.
5. Ripetere tutta la sequenza una seconda volta (sessione B), come controllo.
6. Facoltativo: posizione passeggero, con lo stesso procedimento.

**Percorso trasmesso.** I codec di trasmissione sono non lineari e variano nel tempo. La risposta ricavata dalla registrazione trasmessa è quindi solo **indicativa**: serve soprattutto a stimare la banda e il livello del canale di trasmissione. La risposta di riferimento è quella **locale**.

---

## 7. Deconvoluzione e controllo di qualità

Se possibile si fa sul posto, così una misura difettosa si ripete subito.

```bash
python -m forensicaudio.vehicle.measure deconvolvi V01_20261120_sweep_guidatore_locale_01.wav \
    --parametri sweep/sweep_params.json \
    --out V01_20261120_ir_guidatore_locale_01.wav
```

Il comando:
- scrive la risposta all'impulso (WAV a 32 bit float);
- scrive accanto un file `.json` con lo SHA-256 della registrazione grezza, del file di sweep e della risposta, i parametri usati e gli indicatori di qualità.

La registrazione non deve essere sincronizzata con il lettore: le ripetizioni vengono trovate automaticamente.

Criteri di accettazione:

| Indicatore | Accettabile | Se non lo è |
|---|---|---|
| Ripetizioni trovate | 3/3 | Controllare che la registrazione contenga tutto lo sweep |
| Rapporto picco/rumore | ≥ 40 dB (≥ 30 dB con annotazione) | Alzare il volume, ridurre il rumore ambiente, ripetere |
| Correlazione minima tra ripetizioni | ≥ 0,9 | Rumori o movimenti durante la misura: ripetere |

Opzioni utili:
- `--durata-ir` (predefinita 0,5 s, ampiamente sufficiente in un abitacolo);
- `--pre`: margine prima del picco, predefinito 20 ms;
- `--ripetizioni`.

---

## 8. Registrazioni del rumore per condizione

Ogni condizione si registra per **almeno 2 minuti**, a velocità il più possibile costante, sul percorso locale e, se possibile, anche sul trasmesso.

Durante le registrazioni:
- nessuno parla;
- autoradio spenta;
- niente indicatori di direzione se evitabili;
- annotare con l'orario ogni evento: buche, sorpassi di mezzi pesanti, gallerie, pioggia, clacson. Se qualcuno parla o c'è musica, annotarne l'orario: quei tratti verranno esclusi.

**Condizioni di marcia:**

| Codice | Condizione | Velocità indicativa |
|---|---|---|
| `spento` | Motore spento (quadro acceso se serve la ventilazione) | 0 |
| `minimo` | Motore al minimo, veicolo fermo | 0 |
| `citta` | Città | 30-50 km/h |
| `extraurbano` | Strada extraurbana | 70-90 km/h |
| `autostrada` | Autostrada | 110-130 km/h, entro i limiti |

**Per ogni condizione di marcia**, le combinazioni:
- finestrini chiusi o aperti (`fin-chiusi`, `fin-aperti`; annotare quali finestrini e quanto aperti);
- ventilazione spenta o accesa (`vent-off`, `vent-on`; annotare il livello).

In totale sono 20 registrazioni. Se il tempo è poco, dare priorità a:
1. `minimo`, `citta`, `extraurbano` e `autostrada` con finestrini chiusi e ventilazione spenta;
2. le stesse con ventilazione accesa;
3. poi i finestrini aperti, che pesano molto sull'intelligibilità (Weng et al. 2016).

Annotare anche: tipo di asfalto, meteo, temperatura esterna, tipo di pneumatici (estivi o invernali), marcia inserita e giri motore indicativi.

---

## 9. Metadati

### 9.1 Verbale

Il verbale (cartaceo o digitale firmato) riporta:
- riferimenti del provvedimento;
- data, luogo, orari di inizio e fine;
- operatori;
- veicolo: identificativo interno, marca e modello, anno, alimentazione, pneumatici. La targa va solo nel verbale; nei file si usa l'identificativo interno;
- microfono: modello, numero di serie, firmware, posizione e montaggio, impostazioni (AGC, guadagno, frequenza di campionamento, codec locale); se è un esemplare dello stesso modello, dirlo;
- riproduzione: altoparlante, lettore, livello, SHA-256 del file di sweep;
- posizioni e distanze misurate, riferimenti alle foto;
- l'elenco dei file con il loro SHA-256 (sezione 11);
- anomalie ed eventi.

### 9.2 `misura.json`

Il file di sessione, nella stessa cartella dei file, è letto da `build_profile_from_manifest`. I percorsi sono relativi alla cartella del file. Le chiavi aggiuntive (per esempio `playback`, `ambiente`) vengono conservate nei metadati del profilo.

```json
{
  "vehicle_id": "V01",
  "vehicle": {"marca_modello": "…", "anno": 2019, "alimentazione": "diesel", "pneumatici": "invernali"},
  "date": "2026-11-20",
  "operators": ["operatore A", "operatore B"],
  "microphone": {
    "posizione": "plafoniera, lato guidatore",
    "modello": "…", "numero_serie": "…", "firmware": "…",
    "agc": "non disattivabile", "sample_rate": 16000,
    "esemplare": "dispositivo installato"
  },
  "playback": {"altoparlante": "…", "lettore": "…", "livello_dBA": 80,
               "sweep_sha256": "…"},
  "positions": {"altezza_bocca_da_seduta_cm": 68, "distanza_mic_cm": 55},
  "impulse_responses": [
    {"path": "V01_20261120_ir_guidatore_locale_01.wav", "position": "driver", "channel_path": "local"},
    {"path": "V01_20261120_ir_guidatore_trasmesso_01.wav", "position": "driver", "channel_path": "transmitted",
     "notes": "solo indicativa (codec)"}
  ],
  "noise": [
    {"path": "V01_20261120_rumore_minimo-fin-chiusi-vent-off_locale_01.wav",
     "label": "minimo-fin-chiusi-vent-off", "speed_kmh": 0, "windows_open": false, "ventilation": false,
     "channel_path": "local"},
    {"path": "V01_20261120_rumore_autostrada-fin-chiusi-vent-on_locale_01.wav",
     "label": "autostrada-fin-chiusi-vent-on", "speed_kmh": 120, "windows_open": false, "ventilation": true,
     "channel_path": "local", "notes": "galleria a 01:10-01:25"}
  ],
  "notes": "Sessione B ripetuta per vento forte."
}
```

Valori ammessi:
- `position`: `driver`, `passenger`;
- `channel_path`: `local`, `transmitted`, `unknown`.

---

## 10. Nomi dei file

```
<veicolo>_<AAAAMMGG>_<tipo>_<condizione>_<percorso>_<nn>.<estensione>
```

| Campo | Valori |
|---|---|
| `veicolo` | identificativo interno (es. `V01`) |
| `tipo` | `sweep` = registrazione grezza dello sweep; `ir` = risposta all'impulso calcolata; `rumore` |
| `condizione` | per `sweep`/`ir`: `guidatore`, `passeggero`; per `rumore`: `<marcia>-<finestrini>-<ventilazione>` |
| `percorso` | `locale`, `trasmesso` |
| `nn` | progressivo a due cifre |

Esempi:
- `V01_20261120_sweep_guidatore_locale_01.wav`
- `V01_20261120_ir_guidatore_locale_01.wav` (con accanto `V01_20261120_ir_guidatore_locale_01.json`)
- `V01_20261120_rumore_citta-fin-aperti-vent-off_trasmesso_01.wav`

L'etichetta del rumore in `misura.json` (`label`) coincide con il campo `condizione` del nome del file.

I file grezzi mantengono il formato prodotto dal dispositivo. Non si convertono: la conversione è un'elaborazione e si documenta come tale.

---

## 11. Hash e catena di custodia

1. **Subito dopo lo scarico** dal dispositivo e dalla stazione, calcolare lo SHA-256 di ogni file e salvarlo:
   ```bash
   sha256sum *.wav *.amr *.json > SHA256SUMS
   ```
   Riportare gli hash nel verbale.
2. Fare **due copie**: una master in sola lettura, una di lavoro. Verificare gli hash su entrambe (`sha256sum -c SHA256SUMS`).
3. I file originali **non si modificano mai**: niente tagli, normalizzazioni o conversioni. Ogni elaborazione (per esempio la deconvoluzione) produce un **nuovo** file, con il proprio hash e un JSON che riporta l'hash dell'origine, i parametri e le versioni del software (SWGDE 20-A-001-2.0).
4. Registrare chi ha fatto cosa, quando, dove e con quale strumento.
5. **Import nel caso.** Copia i file in `profiles/vehicle-<id>/` del caso, in sola lettura, verificando l'hash della copia, e salva `vehicle_profile.json`:
   ```python
   from pathlib import Path
   from forensicaudio.case import Case
   from forensicaudio.vehicle.profile import build_profile_from_manifest

   case = Case.open(Path("cases/<id-caso>"))
   profile = build_profile_from_manifest(Path("misura/misura.json"), case=case)
   ```
6. Conservare insieme:
   - file di sweep e `sweep_params.json`;
   - registrazioni grezze;
   - risposte all'impulso e i loro JSON;
   - `misura.json`, `SHA256SUMS`;
   - verbale e foto.

---

## 12. Se non si può misurare

Senza accesso al veicolo si usa il **profilo stimato** (`estimate_vehicle_profile`). Funziona così:
- individua i tratti senza voce dell'**intera** registrazione, con una regola deterministica su energia e piattezza spettrale; i parametri vengono registrati nel profilo;
- ne ricava lo spettro medio del rumore e la distribuzione del rapporto segnale/rumore del caso;
- traccia nel tempo le armoniche del motore;
- salva l'audio di quei tratti come registrazione di rumore utilizzabile dalla simulazione.

Il profilo è marcato `estimated: true`. Non contiene una risposta all'impulso: la simulazione userà solo il rumore, o risposte di altri veicoli dichiarate come tali.

---

## 13. Limiti noti

- L'altoparlante non è una bocca umana: direttività diversa e poca resa sotto i 100-150 Hz.
- Manca il corpo del guidatore, che assorbe e riflette.
- AGC, riduzione del rumore e altre elaborazioni interne del dispositivo alterano la misura.
- Piccole differenze di clock tra lettore e registratore sono tollerate, ma possono attenuare le alte frequenze della risposta.
- La risposta ricavata dal percorso trasmesso è solo indicativa (sezione 6).
- Il rumore cambia con asfalto, meteo, pneumatici e stato del veicolo: annotare tutto, perché la simulazione e l'autovalutazione lavorano per condizione.

---

## 14. Checklist rapida

- [ ] Provvedimento verificato; luogo riservato; due operatori per le prove in marcia
- [ ] Foto dell'abitacolo prima dell'intervento
- [ ] Sweep generato; SHA-256 annotato; effetti audio del lettore disattivati
- [ ] Altoparlante alla bocca del guidatore; distanze misurate; foto
- [ ] Livello regolato senza saturazione; volume poi fisso
- [ ] Sweep guidatore: sessioni A e B (locale e, se possibile, trasmesso)
- [ ] Deconvoluzione sul posto: 3/3 ripetizioni, picco/rumore ≥ 40 dB, correlazione ≥ 0,9
- [ ] Rumori: almeno le 4 condizioni prioritarie, 2 minuti ciascuna, eventi annotati
- [ ] `misura.json` compilato; nomi dei file conformi
- [ ] SHA-256 di tutti i file; due copie verificate
- [ ] Abitacolo ripristinato; foto dopo l'intervento
- [ ] Import nel caso con `build_profile_from_manifest`
