import copy
from aerorozum.requalified_frames import stage


def row(index,status,sha=None):
    return dict(frame_index=index,source={'object':'thermal.avi','generation':'1','video_sha256':'video'},
        sha256=sha or str(index),split='train',airborne_eligible=status=='eligible',
        eligibility=dict(status=status,telemetry={'sha256':'telemetry'},reason='original'))


def test_requalification_adds_only_unknown_or_new_samples_without_rebuy():
    old=[row(0,'eligible'),row(2,'unknown'),row(4,'unknown')];original=copy.deepcopy(old)
    new=[row(0,'eligible'),row(2,'eligible'),row(3,'eligible'),row(4,'eligible')]
    selected,receipt=stage([{'frames':old}],{'frames':new},new,[row(0,'eligible'),row(99,'eligible',sha='4')])
    assert [r['frame_index'] for r in selected]==[2,3]
    assert receipt['excluded']=={'previously_known_frame':1,'existing_high_thinking_pixels':1}
    assert old==original and not receipt['known_decisions_overridden']


def test_any_conflicting_known_decision_quarantines_its_source():
    old=[row(0,'ground'),row(2,'unknown')];new=[row(0,'eligible'),row(2,'eligible')]
    selected,receipt=stage([{'frames':old}],{'frames':new},new,[])
    assert selected==[] and len(receipt['conflicts'])==1


def test_legacy_audit_without_split_requires_the_frozen_registry():
    import pytest
    prior=row(2,'unknown');prior.pop('split');prior['source']['session_id']='flight'
    new=row(2,'eligible');new['source']['session_id']='flight'
    with pytest.raises(ValueError,match='registry'):
        stage([{'frames':[prior]}],{'frames':[new]},[new],[])
    selected,_=stage([{'frames':[prior]}],{'frames':[new]},[new],[],{'flight':'train'})
    assert len(selected)==1
    with pytest.raises(ValueError,match='registry'):
        stage([{'frames':[prior]}],{'frames':[new]},[new],[],{'flight':'validation'})
