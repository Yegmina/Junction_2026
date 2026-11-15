"""Generate the small shell bootstrap, bound to the reviewed Python wizard."""
import hashlib
from pathlib import Path


def render(wizard):
    checksum = hashlib.sha256(Path(wizard).read_bytes()).hexdigest()
    return '''#!/usr/bin/env bash
set -euo pipefail
umask 077

if ! command -v python3 >/dev/null 2>&1; then
  echo "AeroRozum needs Python 3 to start the setup wizard."
  read -r -p "Install Python using apt? [Y/n]: " answer </dev/tty
  case "$answer" in n|N|no|NO) exit 1;; esac
  if ! command -v apt-get >/dev/null 2>&1; then
    echo "Please install Python 3.9+ for your OS and rerun this launcher."
    exit 1
  fi
  if [ "$(id -u)" -eq 0 ]; then
    apt-get update
    apt-get install -y python3 ca-certificates
  else
    sudo apt-get update
    sudo apt-get install -y python3 ca-certificates
  fi
fi
if ! python3 -c 'import sys; sys.exit(sys.version_info < (3, 9))'; then
  echo "The starter needs Python 3.9+ (Ubuntu 22.04+ works). Harbor's Python 3.12 is installed automatically."
  exit 1
fi

starter_tmp=$(mktemp -d)
trap 'rm -rf "$starter_tmp"' EXIT
curl --proto '=https' --tlsv1.2 -fsSL \\
  'https://storage.googleapis.com/aerorozum-junction2026-submission-avid-relic-509408-e4/opencode-wizard-v4.py' \\
  -o "$starter_tmp/wizard.py"
python3 - "$starter_tmp/wizard.py" <<'VERIFY'
import hashlib, pathlib, sys
if hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest() != 'WIZARD_CHECKSUM':
    raise SystemExit('Starter verification failed. Downloaded code was not executed.')
VERIFY
python3 "$starter_tmp/wizard.py" "$@"
'''.replace('WIZARD_CHECKSUM', checksum)


if __name__ == '__main__':
    root = Path(__file__).resolve().parent
    (root / 'start-opencode.sh').write_text(render(root / 'opencode_wizard.py'))
