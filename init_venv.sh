#! /usr/bin/env bash

virtualenv -p python3.8 venv
# /usr/bin/python3 venv
source venv/bin/activate

# install everything
pip install -r requirements.txt

deactivate
