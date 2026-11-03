#!/bin/bash
cd "$(dirname "$0")"
[ -d .venv ] || python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip -q
pip install -r requirements.txt -q
echo "Faerdig! Environment er klar."
