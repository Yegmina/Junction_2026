import json
from pathlib import Path
import pytest
from aerorozum.quarantine_recovery import receipt_records

def test_trailing_nul_cleanup_preserves_every_real_timestamp_and_raw_file(tmp_path):
 rows=[{'frame':1,'received_at_ns':1793801886212254556,'segment':0},{'frame':2,'received_at_ns':1793801886290966814,'segment':0}];raw=('\n'.join(json.dumps(x) for x in rows)+'\n').encode()+b'\0'*3115;source=tmp_path/'source.jsonl';source.write_bytes(raw)
 found,meta=receipt_records(source,tmp_path/'derived.jsonl')
 assert found==rows and source.read_bytes()==raw and meta['trailing_NUL_bytes_removed']==3115
 assert meta['timestamp_values_modified'] is False

def test_malformed_record_inside_stream_is_not_silently_removed(tmp_path):
 p=tmp_path/'source.jsonl';p.write_bytes(b'{"frame":1}\nnot-json\n'+b'\0'*4)
 with pytest.raises(ValueError):receipt_records(p,tmp_path/'derived.jsonl')
 assert not (tmp_path/'derived.jsonl').exists()
