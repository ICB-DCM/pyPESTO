#!/bin/bash
# Run jupyter notebooks as given on command line, show output only on error. 
# If a directory is provided, run all contained notebooks non-recursively.

runNotebook () {
    tempfile=$(tempfile)
    ipython3 nbconvert --debug --stdout --execute --to markdown $@ &> $tempfile
    ret=$?
    if [[ $ret != 0 ]]; then cat $tempfile; fi
    rm $tempfile
    exit $ret
}

if [ $# -eq 0 ]; then
    echo "Usage: $0 [notebook.ipynb] [dirContainingNotebooks/]"
    exit 1
fi

for arg in "$@"; do
    if [ -d $arg ]; then
        for notebook in $(ls -1 $arg | grep -E ipynb\$); do
            runNotebook $arg/$notebook
        done
    elif [ -f $arg ]; then
        runNotebook $arg
    fi
done
exit 0
