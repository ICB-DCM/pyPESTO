#!/bin/sh

python3 -m flake8 \
    --exclude=build,doc,example,tmp,amici_models \
    --per-file-ignores='*/__init__.py:F401'
