#!/bin/bash
# Run jupyter notebook, show output only on error
tempfile=$(tempfile)
ipython3 nbconvert --debug --stdout --execute --to markdown $@ &> $tempfile
ret=$?
if [[ $ret != 0 ]]; then cat $tempfile; fi
rm $tempfile
exit $ret
