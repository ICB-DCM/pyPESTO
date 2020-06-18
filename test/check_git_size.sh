#!/bin/sh

size=$(du -s .git | cut -f1)

if [ "$size" -gt 20000 ]; then
  echo "Git history is suspiciously large: $size"
fi
