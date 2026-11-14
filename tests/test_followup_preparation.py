import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('followup', Path(__file__).resolve().parents[1] / 'cloud/prepare_followup_sources.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def receipts():
    return [{'segment': 3, 'frame': 10 + i, 'received_at_ns': 1000000000 + i * 40000000} for i in range(3)]


def test_original_integer_receipts_are_preserved_and_other_segments_excluded():
    original = receipts()
    joined = module.aligned_segment([{'segment': 2}, *original], 3, [0, .04, .08])
    assert joined == original
    assert all(a is b for a, b in zip(joined, original))


@pytest.mark.parametrize('change,pts,reason', [
    (lambda r: r.pop(), [0, .04, .08], 'count mismatch'),
    (lambda r: r[1].update(frame=99), [0, .04, .08], 'not contiguous'),
    (lambda r: r[1].update(received_at_ns=r[0]['received_at_ns']), [0, .04, .08], 'Receipt timestamps'),
    (lambda r: r[1].pop('frame'), [0, .04, .08], 'counters are required'),
    (lambda r: None, [0, float('nan'), .08], 'Native presentation'),
    (lambda r: None, [0, .04, .04], 'Native presentation'),
])
def test_ambiguous_clock_alignment_is_quarantined(change, pts, reason):
    original = receipts()
    change(original)
    with pytest.raises(ValueError, match=reason):
        module.aligned_segment(original, 3, pts)


def test_missing_host_decoder_fails_before_reading_sources_or_creating_outputs(monkeypatch):
    monkeypatch.setattr(module.shutil, 'which', lambda _: None)
    with pytest.raises(RuntimeError, match='host dependencies missing: ffprobe, ffmpeg'):
        module.prepare(None)
