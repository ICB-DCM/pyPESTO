#!/bin/bash
# Run jupyter notebook, show output only on error
tempfile=$(tempfile)
jupyter nbconvert --debug --stdout --execute --to markdown $@ &> $tempfile
ret=$?
if [[ $ret != 0 ]]; then cat $tempfile; fi
rm $tempfile
exit $ret
