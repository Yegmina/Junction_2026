#!/bin/bash
set -euo pipefail
/usr/bin/chown root:root /logs
/usr/bin/chmod 755 /logs
if [ -L /logs/verifier ]; then /usr/bin/rm /logs/verifier; fi
/usr/bin/mkdir -p /logs/verifier
chown root:root /logs/verifier
chmod 700 /logs/verifier
find /logs/verifier -mindepth 1 -maxdepth 1 -exec rm -rf -- {} +
/usr/bin/python3 /tests/verify.py
