"""Build a checked candidate arrival catalog without changing running training."""
import argparse
import json
from aerorozum.vm_ingestion import build_candidate

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labeler-root', default='/opt/aerorozum/airborne-labeler/artifacts')
    parser.add_argument('--trainer-root', default='/opt/aerorozum/artifacts')
    args = parser.parse_args()
    print(json.dumps(build_candidate(args.labeler_root, args.trainer_root), indent=2), flush=True)
