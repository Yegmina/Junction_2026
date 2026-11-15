import pytest
from aerorozum.receipt_clock import rollback_windows,safe_native_indices


def records(times):return [dict(frame=i+1,received_at_ns=t*1_000_000_000) for i,t in enumerate(times)]


def test_rollback_overlap_and_full_freshness_margin_are_excluded():
    rows=records([100,101,102,103,104,103,104,105,106,107,108])
    indices,proof=safe_native_indices(rows,len(rows),[])
    assert indices==[0,9,10] and proof['rollback_windows_ns']==[(101_000_000_000,106_000_000_000)]
    assert not proof['timestamps_modified']


def test_telemetry_rollbacks_also_block_apparently_monotonic_thermal_frames():
    rows=records(list(range(100,111)))
    safe,proof=safe_native_indices(rows,len(rows),[[100_000_000_000,106_000_000_000,104_000_000_000,110_000_000_000]])
    assert safe==[0,1,9,10]


def test_no_clock_recovery_guesses_missing_frame_identity():
    rows=records([100,101,102])
    with pytest.raises(ValueError,match='count mismatch'):safe_native_indices(rows,2,[])
    rows[1]['frame']=99
    with pytest.raises(ValueError,match='Contiguous'):safe_native_indices(rows,3,[])
    with pytest.raises(ValueError,match='two-second'):rollback_windows([[1,2]],guard_ns=0)


def test_duplicates_do_not_inflate_sampling_rate_and_inputs_are_unchanged():
    rows=records([100,101,101,102]);saved=[dict(r) for r in rows]
    selected,_=safe_native_indices(rows,4,[])
    assert selected==[0,1,3] and rows==saved
