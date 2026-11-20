# Contributing

Original project code uses [SANSEVIERIA License 1.1](LICENSE), except for the separately licensed
material listed in [NOTICE](NOTICE). Contributors retain their copyright; no assignment is
requested. Preserve third-party notices.

## Licence assent for contributions under SANSEVIERIA License

Before submitting code covered by SANSEVIERIA License 1.1, read the complete licence. Include this
express declaration in your PR or an associated signed record, identifying the contribution and
licence version:

> I intentionally submit the identified Contribution under the SANSEVIERIA License 1.1, whose complete text has been provided to me. I have the authority necessary to make the copyright and patent grants applicable to this Contribution. I expressly agree to the limited authority in section 7 allowing SANSEVIERIA OÜ to authorize specified commercial offerings and charge for those authorizations while preserving the source, reciprocity, notice and recipient-rights conditions, and to the authority in section 15 to make the Contribution available under later common versions that preserve the guarantees listed there. I understand that both authorities are personal to SANSEVIERIA OÜ, end or are suspended as provided in section 7, including on a Change of Control or its dissolution, and can be renewed only by my express statement after the event. I retain my rights and do not grant a general proprietary relicensing power. I have identified separately licensed material and other rights holders whose permission is relevant, and I do not purport to grant rights they have not authorized me to grant.

This declaration is specific to SANSEVIERIA License 1.1, not the standard DCO. Maintainers must
record the assent and verify the scope of any employer or third-party permission. An automatically
filled checkbox, silence or merely opening a PR does not establish that assent.

SANSEVIERIA License is a single licence for all SANSEVIERIA OÜ projects. Section 15 lets SANSEVIERIA
OÜ move contributions, and changes published in forks, to later common versions that keep the
guarantees it lists: free use, source availability, reciprocity, the patent permission,
contributors' rights and the limits on SANSEVIERIA OÜ's authority. It allows no proprietary
relicensing, and recipients of earlier copies keep the rights they received.

Under section 7, the authorities you grant SANSEVIERIA OÜ cannot be transferred. They end for your
code if someone else acquires control of the company or it is dissolved or declared bankrupt, and
are suspended during insolvency proceedings; only your express statement after the event renews
them. Your rights and every recipient's rights stay unchanged.

SANSEVIERIA OÜ and its appointed maintainers decide which contributions to merge and which releases
are official. Submission does not guarantee acceptance.

## Layout

| Path | Contents |
| --- | --- |
| `src/akoe/` | The engine: cases and chain of custody (`case.py`, `models.py`), workflows (`engine.py`), API (`api/`), CLI (`cli.py`) |
| `src/akoe/speaker/` | Subject model: dataset, timbre, voiceprint, calibration, verification loop |
| `src/akoe/vehicle/` | Vehicle profile, measurement tools, channel and codec simulation, local/transmitted pairs |
| `src/akoe/stages/` | Audio stages: preprocessing, cleaning for listening, VAD |
| `src/akoe/transcribe/` | Transcription backends, alternatives, guard, candidate scoring, alignment |
| `src/akoe/adapt/` | Subject model adaptation (LoRA/full), adapters, case lexicon |
| `src/akoe/assess/` | Metrics and per-case self-assessment |
| `docs/` | Runbooks and decision notes (Italian) |

## Build and test

```sh
python -m venv .venv && .venv/bin/pip install -e ".[ml,assess,codecs,api,dev]"
.venv/bin/python -m pytest -q            # fast tests, no downloads
.venv/bin/python -m pytest -q -m slow    # real models, downloads them
```

Every source file starts with `SPDX-License-Identifier: LicenseRef-SANSEVIERIA-1.1`. Never commit
audio, case folders, model weights or credentials. Read `CLAUDE.md` for the invariants.
