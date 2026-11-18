#!/bin/sh

# Verify that every test/<dir> containing tests is referenced somewhere in
# tox.ini, so a new test directory can't silently go unrun by every CI job.

missing=""

for dir in test/*/; do
  dir="${dir%/}"
  # skip directories with no test_*.py files anywhere inside
  if ! find "$dir" -name 'test_*.py' | grep -q .; then
    continue
  fi
  if ! grep -qF "$dir" tox.ini; then
    missing="$missing $dir"
  fi
done

if [ -n "$missing" ]; then
  echo "The following test directories are not referenced in tox.ini and will never run in CI:" >&2
  for d in $missing; do
    echo "  $d" >&2
  done
  exit 1
fi

echo "All test directories with test_*.py files are referenced in tox.ini."
