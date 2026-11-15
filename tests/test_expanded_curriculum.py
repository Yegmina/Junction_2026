import pytest
from aerorozum.expanded_curriculum import select


def row(name, positive=False, **extra):
    return dict(id=name,sha256=name,split='train',training_eligible=True,provenance='user_accepted_model',
                flight_state='airborne',source_group='s',boxes=[{'bbox':{}}] if positive else [],
                detections=[{}] if positive else [],zero_candidates=not positive,**extra)


def test_expansion_keeps_all_positives_and_blocks_test_ground_uncertain():
    ground=row('ground',True);ground['flight_state']='known_ground'
    held=row('held',True)
    uncertain=row('uncertain',True,uncertain_boxes=[{}])
    catalog=dict(images=[row('p1',True),row('p2',True),row('n1'),row('n2'),row('n3'),held,ground,uncertain],
                 dataset_signature='frozen',session_splits={},sealed_test_image_hashes=['held'])
    result=select(catalog)
    assert len(result['images'])==4
    assert {r['id'] for r in result['images'] if r['boxes']}=={'p1','p2'}
    assert result==select(catalog)
    assert not result['metadata']['test_accessed']


def test_no_invented_backgrounds_or_negative_shortage():
    bad=row('bad');bad['detections']=[{'confidence':.2}]
    with pytest.raises(ValueError,match='explicit zero'):
        select(dict(images=[row('p',True),bad],dataset_signature='frozen',session_splits={}))
