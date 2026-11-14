#!/bin/bash
set -euo pipefail
mkdir -p /logs/verifier
chown root:root /logs/verifier
chmod 700 /logs/verifier
find /logs/verifier -mindepth 1 -maxdepth 1 -exec rm -rf -- {} +
python3 /tests/verify.py
