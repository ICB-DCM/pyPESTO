#!/bin/sh

# Check .git log size
# Over time, this will increase s.t. the value will need to be adjusted.
# However, if jumps in the size occur, this may indicate that undesirably
# large files have been included.

size=$(du -s .git | cut -f1);
# reference: 45140 2020-06-19

if [ "$size" -gt 50000 ]; then
  echo "Git history is suspiciously large: $size";
  exit 1;
fi
