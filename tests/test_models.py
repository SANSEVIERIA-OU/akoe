# SPDX-License-Identifier: LicenseRef-SANSEVIERIA-1.1
import pytest
from pydantic import ValidationError

from akoe.config import Channel, RecordingSource, StageKind, VocalEffort
from akoe.models import (
    AdapterArtifact,
    AdapterTechnique,
    Alternative,
    ArtifactRef,
    HypothesisTranscript,
    RecordingCondition,
    SegmentFlag,
    SegmentHypotheses,
    SelfAssessment,
    SpeakerProfile,
    ConditionProfile,
    VehicleProfile,
    WordHypothesis,
)


def test_segment_hypotheses_requires_provenance():
    with pytest.raises(ValidationError):
        SegmentHypotheses(start=0.0, end=1.0)  # type: ignore[call-arg]


def test_hypothesis_transcript_is_always_ai_hypothesis():
    t = HypothesisTranscript(case_id="c1")
    assert t.is_ai_hypothesis is True
    with pytest.raises(ValidationError):
        HypothesisTranscript(case_id="c1", is_ai_hypothesis=False)


def test_alternative_requires_origin():
    with pytest.raises(ValidationError):
        Alternative(text="ciao")  # type: ignore[call-arg]


def test_transcript_round_trip():
    seg = SegmentHypotheses(
        start=0.0,
        end=2.0,
        audio_provenance=StageKind.NEURAL,
        hypotheses=[
            Alternative(text="vado a casa", score=-1.2, origin="whisper/generic"),
            Alternative(text="vado a cassa", score=-1.9, origin="parakeet/adapted:ab12"),
        ],
        words=[WordHypothesis(text="casa", start=1.0, end=1.5, confidence=0.6, origin="whisper/generic",
                              alternatives=[Alternative(text="cassa", origin="parakeet/adapted:ab12")])],
        flags=[SegmentFlag.MODEL_DISAGREEMENT],
    )
    t = HypothesisTranscript(case_id="c1", segments=[seg], models=["whisper", "parakeet"])
    back = HypothesisTranscript.model_validate_json(t.model_dump_json())
    assert back == t


def test_condition_key_is_stable():
    c = RecordingCondition(channel=Channel.CAR, source=RecordingSource.TRANSMITTED,
                           codec="amr-nb", vocal_effort=VocalEffort.LOW)
    assert c.key() == "car/transmitted/amr-nb/low"
    assert RecordingCondition().key() == "unknown/other/unknown-codec/unknown"


def test_artifact_ref_verify(tmp_path):
    f = tmp_path / "profile.json"
    f.write_text("{}")
    ref = ArtifactRef.from_file(f, kind="speaker_profile", root=tmp_path)
    assert ref.path == "profile.json"
    assert ref.verify(tmp_path)
    f.write_text('{"tampered": true}')
    assert not ref.verify(tmp_path)


def test_profiles_and_adapter_round_trip(tmp_path):
    w = tmp_path / "adapter.bin"
    w.write_bytes(b"weights")
    ref = ArtifactRef.from_file(w, kind="adapter", root=tmp_path)
    adapter = AdapterArtifact(
        adapter_id="subj1-v1", base_model="openai/whisper-large-v3", technique=AdapterTechnique.LORA,
        hyperparams={"r": 32}, seed=1234, train_manifest_sha256="0" * 64,
        conditions=["car/local/unknown-codec/normal"], weights=ref,
    )
    assert AdapterArtifact.model_validate_json(adapter.model_dump_json()) == adapter

    sp = SpeakerProfile(subject_id="subj1", conditions=[ConditionProfile(condition=RecordingCondition(channel=Channel.PHONE))])
    assert SpeakerProfile.model_validate_json(sp.model_dump_json()) == sp

    vp = VehicleProfile(vehicle_id="veh1", estimated=True, observed_codecs=["opus"])
    assert VehicleProfile.model_validate_json(vp.model_dump_json()) == vp

    sa = SelfAssessment(case_id="c1", overall={"wer": 0.31}, limitations=["held-out is not self-talk"])
    assert SelfAssessment.model_validate_json(sa.model_dump_json()) == sa
