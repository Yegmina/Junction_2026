from collections import Counter
from copy import deepcopy
import pytest
from aerorozum.replay_curriculum import repeat_positives


def samples():return [{'id':'p0','gold_boxes':[{'bbox':'unchanged'}]},{'id':'n0','gold_boxes':[]},{'id':'p1','gold_boxes':[{'bbox':'also unchanged'}]},{'id':'n1','gold_boxes':[]}]


def test_factor_one_preserves_exact_order_and_object_identity():
    rows=samples();result,protocol=repeat_positives(rows,1)
    assert result==rows and all(a is b for a,b in zip(rows,result))
    assert protocol['positive_entry_fraction']==.5


def test_four_preserves_every_example_and_yields_eighty_percent_positive():
    rows=samples();before=deepcopy(rows);result,protocol=repeat_positives(rows,4)
    assert Counter(r['id'] for r in result)=={'p0':4,'p1':4,'n0':1,'n1':1}
    assert protocol['positive_entry_fraction']==.8 and rows==before
    assert protocol['every_source_example_retained'] is True


@pytest.mark.parametrize('factor',[0,2,True,4.0])
def test_undeclared_factors_rejected(factor):
    with pytest.raises(ValueError):repeat_positives(samples(),factor)


def test_missing_class_rejected():
    with pytest.raises(ValueError):repeat_positives([{'gold_boxes':[]}],4)
