#!/usr/bin/env python3
"""Download pinned, unmodified DuMux/DUNE sources and run upstream evaporation.

Builds and vendor files belong on the external SSD, never in the source tree.
Retains upstream licensing. This script records actual results, never assumes
that downloading or compiling a model establishes physical validation.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/media/yehormsi/External SSD Volume/AeroRozum-thermal-rl/physics/dumux'))
    p.add_argument('--jobs',type=int,default=4)
    args=p.parse_args();root=args.root;root.mkdir(parents=True,exist_ok=True)
    # DUNE-generated CMake package files do not safely quote space-containing
    # prefixes. A stable symlink keeps all bytes on the SSD with a safe prefix.
    if ' ' in str(root):
        alias=Path('/tmp')/('aerorozum-dumux-'+hashlib.sha256(str(root).encode()).hexdigest()[:12])
        if not alias.exists():alias.symlink_to(root,target_is_directory=True)
        if alias.resolve()!=root.resolve():raise RuntimeError('Build alias points to another directory')
        root=alias
    log=root/'build.log';status=root/'status.json'
    state=dict(solver='DuMux',version='3.10.0',dune_version='2.10.0',status='running',research_accepted=False,started_at=time.time(),sources=[])
    def save():status.write_text(json.dumps(state,indent=2))
    def run(command,cwd):
        print('Running',command,flush=True)
        with log.open('a') as out:
            out.write('\n$ '+repr(command)+'\n');out.flush()
            subprocess.run(command,cwd=cwd,stdout=out,stderr=subprocess.STDOUT,check=True)
    try:
        save()
        for module in ['dune-common','dune-geometry','dune-grid','dune-istl','dune-localfunctions','dumux']:
            version='3.10.0' if module=='dumux' else '2.10.0'
            url=f'https://codeload.github.com/dumux/dumux/tar.gz/refs/tags/{version}' if module=='dumux' else f'https://gitlab.dune-project.org/core/{module}/-/archive/v{version}/{module}-v{version}.tar.gz'
            archive=root/f'{module}-{version}.tar.gz';dest=root/module
            if not archive.exists():
                temporary=archive.with_suffix('.download')
                subprocess.run(['curl','--fail','--location','--retry','2','--max-time','180',url,'--output',str(temporary)],check=True)
                temporary.replace(archive)
            checksum=hashlib.sha256(archive.read_bytes()).hexdigest()
            if not dest.exists():
                with tarfile.open(archive) as tar:
                    # No links or path traversal from downloaded archives.
                    base=tar.getmembers()[0].name.split('/')[0]
                    tar.extractall(root,filter='data')
                (root/base).rename(dest)
            state['sources'].append(dict(module=module,version=version,url=url,sha256=checksum));save()
        flags=['-DCMAKE_BUILD_TYPE=Release','-DDUNE_ENABLE_PYTHONBINDINGS=OFF','-DCMAKE_DISABLE_FIND_PACKAGE_LaTeX=ON','-DDUNE_MAX_TEST_CORES=1']
        for module in ['dune-common','dune-geometry','dune-grid','dune-istl','dune-localfunctions']:
            src=root/module;build=src/'build-reference'
            dependency_flags=[f'-D{m}_DIR={root/m/"build-reference"}' for m in ['dune-common','dune-geometry','dune-grid','dune-istl','dune-localfunctions'] if m!=module and (root/m/'build-reference').exists()]
            run(['cmake','-S',str(src),'-B',str(build),*flags,*dependency_flags],root)
            run(['cmake','--build',str(build),'-j',str(args.jobs)],root)
        src=root/'dumux';build=src/'build-reference'
        deps=[f'-D{m}_DIR={root/m/"build-reference"}' for m in ['dune-common','dune-geometry','dune-grid','dune-istl','dune-localfunctions']]
        run(['cmake','-S',str(src),'-B',str(build),*flags,*deps],root)
        run(['cmake','--build',str(build),'--target','test_richardsni_evaporation_tpfa','-j',str(args.jobs)],root)
        run(['ctest','--test-dir',str(build),'-R','^test_richardsni_evaporation_tpfa$','--output-on-failure'],root)
        state.update(status='upstream_test_passed',finished_at=time.time(),validation_state='upstream_regression_verified',limitations=['Only the unmodified upstream evaporation regression was run.','No custom 3D buried-object campaign or field validation is implied.'])
        save();print(json.dumps(state,indent=2));return 0
    except Exception as exc:
        state.update(status='failed',finished_at=time.time(),error=str(exc),log_path=str(log));save();raise


if __name__=='__main__':raise SystemExit(main())
