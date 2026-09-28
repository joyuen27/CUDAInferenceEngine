#!/bin/bash
[ -f "$1" ] || { echo "usage: check-binary.sh <binary>"; exit 1; }
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
cuobjdump --list-elf "$1" | grep -q "sm_${CC}" \
  && echo "OK: sm_${CC} cubin present" \
  || echo "Warn: Rebuild with -gencode=arch=compute_${CC},code=sm_${CC}"
