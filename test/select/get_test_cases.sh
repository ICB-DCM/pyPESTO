#!/bin/bash
if [ ! -d petab_select ]
then
    git clone -n --depth=1 --filter=tree:0 https://github.com/PEtab-dev/petab_select.git
    cd petab_select
    git sparse-checkout set --no-cone /test_cases
    git checkout
fi
