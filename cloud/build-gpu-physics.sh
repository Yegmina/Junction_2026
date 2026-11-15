#!/bin/bash
# Isolated solver-library build; does not alter the CPU solver or Python venv.
set -euo pipefail
root=/opt/aerorozum/gpu-physics
revision=91a8413ef267b1c32aff4014c02820e1c5897ac2
destination=gs://aerorozum-thermal-rl-avid-relic-509408-e4/physics/gpu-backend-build
mkdir -p "$root"
exec > >(tee -a "$root/build.log") 2>&1
finish() {
    code=$?
    trap - EXIT
    /usr/bin/python3 - "$root" "$revision" "$code" <<'PY'
import datetime,json,pathlib,sys
root=pathlib.Path(sys.argv[1]);code=int(sys.argv[3])
libraries=list((root/'AMGX/build').rglob('libamgxsh.so*'))
(root/'build-status.json').write_text(json.dumps({'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source':'https://github.com/NVIDIA/AMGX','revision':sys.argv[2],'exit_code':code,'status':'built' if code==0 and libraries else 'failed','libraries':[str(p) for p in libraries],'cuda_architecture':'80','physics_backend_integrated':False,'numerical_parity_passed':False},indent=2))
PY
    gcloud storage cp "$root/build-status.json" "$root/build.log" "$destination/" --quiet || true
    exit "$code"
}
trap finish EXIT
if ! test -d "$root/AMGX/.git"; then
    git init "$root/AMGX"
    git -C "$root/AMGX" remote add origin https://github.com/NVIDIA/AMGX.git
fi
git -C "$root/AMGX" fetch --depth=1 origin "$revision"
git -C "$root/AMGX" checkout --detach FETCH_HEAD
test "$(git -C "$root/AMGX" rev-parse HEAD)" = "$revision"
if ! test -x "$root/build-tools/bin/cmake"; then
    /usr/bin/python3 -m venv "$root/build-tools"
    "$root/build-tools/bin/pip" install 'cmake>=3.31,<4'
fi
export PATH="$root/build-tools/bin:/usr/local/cuda/bin:$PATH"
cmake -S "$root/AMGX" -B "$root/AMGX/build" \
    -DCMAKE_BUILD_TYPE=Release -DCUDA_ARCH=80 \
    -DCMAKE_CUDA_ARCHITECTURES=80 -DAMGX_WITH_MPI=OFF \
    -DCMAKE_DISABLE_FIND_PACKAGE_MPI=TRUE -DCMAKE_NO_MPI=TRUE \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
cmake --build "$root/AMGX/build" --target amgxsh -j16
