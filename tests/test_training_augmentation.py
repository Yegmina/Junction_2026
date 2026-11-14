import hashlib
import numpy as np
from PIL import Image
import pytest
from aerorozum.training_augmentation import augment_training_rows,flip_box


def example(tmp_path):
    pixels=np.arange(36,dtype=np.uint8).reshape(3,4,3)
    source=tmp_path/'frame.png';Image.fromarray(pixels).save(source)
    return dict(split='train',zone='zone1',image_path=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        provenance='original_manual',gold_boxes=[dict(x_min=100,y_min=200,x_max=400,y_max=600)]),pixels


def test_derivatives_transform_pixels_and_boxes_together_and_keep_original_provenance(tmp_path):
    row,pixels=example(tmp_path);original=dict(row)
    rows,receipt=augment_training_rows([row],tmp_path/'cache')
    assert len(rows)==4 and receipt['independent_observations']==1
    assert row==original and all(r['provenance']=='original_manual' for r in rows)
    np.testing.assert_array_equal(np.asarray(Image.open(rows[1]['image_path'])),pixels[:,::-1])
    np.testing.assert_array_equal(np.asarray(Image.open(rows[2]['image_path'])),pixels[::-1])
    np.testing.assert_array_equal(np.asarray(Image.open(rows[3]['image_path'])),pixels[::-1,::-1])
    assert rows[1]['gold_boxes']==[dict(x_min=600,y_min=200,x_max=900,y_max=600)]
    assert rows[3]['gold_boxes']==[dict(x_min=600,y_min=400,x_max=900,y_max=800)]
    assert augment_training_rows([row],tmp_path/'cache')[1]==receipt


@pytest.mark.parametrize('changes',[dict(split='validation'),dict(zone='zone7'),dict(zone=None,parent_zone='zone9')])
def test_held_out_sources_are_rejected_before_materialization(tmp_path,changes):
    row,_=example(tmp_path);row.update(changes)
    with pytest.raises(ValueError,match='Only training'):augment_training_rows([row],tmp_path/'cache')
    assert not (tmp_path/'cache').exists()


def test_tampered_derivative_and_source_are_rejected(tmp_path):
    row,_=example(tmp_path);rows,_=augment_training_rows([row],tmp_path/'cache')
    from pathlib import Path
    Path(rows[1]['image_path']).write_bytes(b'changed')
    with pytest.raises(ValueError,match='cache checksum'):augment_training_rows([row],tmp_path/'cache')
    Path(row['image_path']).write_bytes(b'changed')
    with pytest.raises(ValueError,match='source checksum'):augment_training_rows([row],tmp_path/'new-cache')


def test_box_flip_is_an_involution_including_image_edges():
    box=dict(x_min=0,y_min=100,x_max=1000,y_max=700)
    assert flip_box(flip_box(box,True,True),True,True)==box
