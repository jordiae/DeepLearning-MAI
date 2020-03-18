#!/usr/bin/env bash
src_path="$(realpath ../../src)"
export PYTHONPATH="${PYTHONPATH}:${src_path}"
python3.7 -m venv venv
python3.7 -m pip install -r requirements.txt