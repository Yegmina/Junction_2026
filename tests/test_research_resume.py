import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('resume_research',Path(__file__).resolve().parents[1]/'cloud/resume_research_queue.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_resume_preserves_recipes_gates_and_inference_reservation():
    settings={'sft_steps':32,'grpo_steps':16,'maximum_synthetic_fraction':.25,'seeds':[17,43,91],'deadline_utc':'old'}
    controls={'pause_dispatch':True,'pause_reason':'submission','external_gpu_leases':[{'gpu_ids':[0]}]}
    updated,resumed=module.transition(settings,controls,{'jobs':[{'status':'queued'}]},'2026-11-16T16:46:54+00:00')
    assert {k:v for k,v in updated.items() if k!='deadline_utc'}=={k:v for k,v in settings.items() if k!='deadline_utc'}
    assert resumed['external_gpu_leases']==controls['external_gpu_leases'] and not resumed['pause_dispatch']
    assert controls['pause_dispatch'] is True and 'pause_reason' in controls


def test_resume_refuses_live_training_or_ambiguous_deadline():
    with pytest.raises(ValueError,match='worker is running'):
        module.transition({},dict(pause_dispatch=True),{'evaluation_jobs':[{'status':'running'}]},'2026-11-16T16:46:54+00:00')
    with pytest.raises(ValueError,match='UTC-offset'):
        module.transition({},dict(pause_dispatch=True),{},'2026-11-16T16:46:54')
