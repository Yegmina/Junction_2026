#!/usr/bin/env python3
"""Build and execute pinned DuMux analytical water/heat reference regressions."""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import subprocess
import time
import xml.etree.ElementTree as ET

import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/media/yehormsi/External SSD Volume/AeroRozum-thermal-rl/physics/dumux'))
    p.add_argument('--jobs',type=int,default=4)
    args=p.parse_args();real=args.root
    root=Path('/tmp')/('aerorozum-dumux-'+hashlib.sha256(str(real).encode()).hexdigest()[:12])
    build=root/'dumux/build-reference'
    status=real/'analytical-status.json';log=real/'analytical-build-test.log'
    result=dict(status='running',solver='DuMux',version='3.10.0',started_at=time.time(),research_accepted=False)
    status.write_text(json.dumps(result,indent=2))
    env=os.environ.copy();env['DUMUX_NUM_THREADS']='2'
    targets=['test_richards_analytical_tpfa','test_richardsni_conduction_tpfa','test_richards_benchmark_tpfa']
    try:
        with log.open('a') as out:
            for target in targets:
                subprocess.run(['cmake','--build',str(build),'--target',target,'-j',str(args.jobs)],env=env,stdout=out,stderr=subprocess.STDOUT,check=True)
            subprocess.run(['ctest','--test-dir',str(build),'-R','^(test_richards_analytical_tpfa|test_richardsni_conduction_tpfa)$','--output-on-failure'],env=env,stdout=out,stderr=subprocess.STDOUT,check=True)
        # Publish the actual error against the exact-temperature field rather
        # than calling a numerical-reference comparison exact agreement.
        vtu=build/'test/porousmediumflow/richards/nonisothermal/conduction/test_richardsni_conduction_tpfa-00024.vtu'
        piece=ET.parse(vtu).find('.//Piece')
        arrays={n.get('Name'):np.fromstring(n.text or '',sep=' ') for n in piece.findall('./CellData/DataArray')}
        numerical=arrays.get('T',arrays.get('temperature'))
        exact=arrays['temperatureExact'];error=np.abs(numerical-exact)
        test_log=(build/'Testing/Temporary/LastTest.log').read_text()
        l2=[float(v) for v in re.findall(r'L2 error for\s+\d+ elements:\s*([0-9.eE+-]+)',test_log)]
        result.update(status='upstream_analytical_regressions_passed',completed_at=time.time(),conduction_max_error_k=float(error.max()),conduction_rms_error_k=float(np.sqrt(np.mean(error**2))),conduction_rms_relative_to_temperature_span=float(np.sqrt(np.mean(error**2))/max(np.ptp(exact),1e-12)),richards_manufactured_solution_max_l2=max(l2) if l2 else None,test_names=targets[:2],benchmark_binary_built=True,limitations=['Upstream analytic/regression cases verify components; custom application convergence and budget gates remain separate.'])
        status.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
    except Exception as exc:
        result.update(status='failed',error=str(exc),completed_at=time.time(),log_path=str(log));status.write_text(json.dumps(result,indent=2));raise


if __name__=='__main__':main()
