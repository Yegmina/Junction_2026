"""Index actual saved label prompts without changing old records or calling AI."""
import argparse,json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from aerorozum.common import data_root,save,utcnow,digest
from aerorozum.label_prompts import PROMPT_V1,prompt_metadata,result_prompt_metadata,registry
from aerorozum.labelling import inference_cache_key,MODEL


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root');parser.add_argument('--publish',action='store_true');args=parser.parse_args()
    root=Path(args.root or data_root());folder=root/'labelling/prompt-history';rows=[];counts=Counter()
    provider=SimpleNamespace(cfg={'reserve_output_tokens':2048,'thinking_budget':0,'api_version':'v1beta'})
    for name in ('robotics-er2-5fps','robotics-er2-airborne-priority','robotics-er2-airborne'):
        for path in (root/'labelling'/name/'labels/results').glob('*.json'):
            try:record=json.loads(path.read_text())
            except (OSError,ValueError):continue
            if record.get('model')!=MODEL:continue
            meta=result_prompt_metadata(record);basis='saved_response_or_explicit_hash'
            if meta['label_prompt_version']=='unknown' and record.get('sha256'):
                old_keys={inference_cache_key(record,record['sha256'],provider,prompt=PROMPT_V1),digest([record.get('id'),record['sha256'],MODEL,.6,PROMPT_V1])}
                if path.stem in old_keys:meta=prompt_metadata(PROMPT_V1);basis='verified_original_prompt_cache_identity'
            counts[meta['label_prompt_version']]+=1
            rows.append({'id':record.get('id'),'cache_key':path.stem,'source':record.get('source'),'frame_index':record.get('frame_index'),'image_sha256':record.get('sha256'),'labelled_at':record.get('labelled_at'),'record_status':record.get('status','completed'),**meta,'attribution_basis':basis,'cloud_uris':record.get('cloud_sync',{}).get('uris',{}),'local_result':str(path.relative_to(root))})
    release=json.loads((folder/'release.json').read_text()) if (folder/'release.json').exists() else {}
    stamp=utcnow();catalog={**registry(),'recorded_at':stamp,'deployment_status':release.get('status','candidate_evaluation_before_airborne_release'),'selection':release}
    index={'recorded_at':stamp,'counts_by_prompt':dict(counts),'records':rows,'old_paid_records_modified':False,'new_model_requests':0}
    save(folder/'registry.json',catalog);save(folder/'index.json',index)
    report={'recorded_at':stamp,'counts_by_prompt':dict(counts),'records':len(rows),'index':'labelling/prompt-history/index.json','registry':'labelling/prompt-history/registry.json'}
    if args.publish:
        from aerorozum.cloud_cache import GCSRecordStore
        store=GCSRecordStore('aerorozum-jetson-testdata-smart-bird')
        report['cloud_registry']=store.put(digest(['prompt-registry',catalog]),catalog,kind='progress')
        report['cloud_index']=store.put(digest(['prompt-history-index',index]),index,kind='progress')
    save(folder/'summary.json',report);print(json.dumps(report,indent=2))


if __name__=='__main__':main()
