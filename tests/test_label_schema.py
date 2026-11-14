import copy
import pytest
from aerorozum.label_schema import normalize_thermal_output
from aerorozum.labelling import classify_response


def test_coordinate_aliases_preserve_numbers_confidence_and_raw_output():
    raw={'detections':[{'confidence':.35,'bbox':{'xmin':12,'ymin':23,'xmax':34,'ymax':45}}]}
    original=copy.deepcopy(raw);normalized,changes=normalize_thermal_output(raw)
    assert raw==original and len(changes)==4
    assert normalized['detections'][0]['bbox']==dict(x_min=12,y_min=23,x_max=34,y_max=45)
    classified=classify_response(normalized)
    assert len(classified['uncertain'])==1 and not classified['zero_candidates']


def test_ambiguous_key_is_never_guessed_or_converted_to_background():
    raw={'detections':[{'confidence':.35,'bbox':{'x_min':12,'y_min':23,'x_max':34,'max':45}}]}
    normalized,changes=normalize_thermal_output(raw)
    assert changes==[]
    with pytest.raises(ValueError):classify_response(normalized)


def test_conflicting_aliases_are_rejected():
    with pytest.raises(ValueError,match='Conflicting'):
        normalize_thermal_output({'detections':[{'bbox':{'xmin':12,'x_min':13}}]})
