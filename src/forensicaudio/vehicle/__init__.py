"""Vehicle profile and channel simulation (IMPLEMENTATION_PLAN M3).

- measure:  exponential-sweep impulse response measurement (runbook in docs/)
- profile:  VehicleProfile from measurements, or estimated from a recording
- simulate: delay-compensated IR + real noise at sampled SNR, reproducible
- codecs:   variable-codec augmentation and codec detection
- pairs:    local/transmitted alignment and transmission-channel estimate
"""
