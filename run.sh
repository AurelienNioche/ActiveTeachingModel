#!/bin/bash
makedir -p triton_out
truncate --size 0 triton_out/debug.out
echo -n $@
PID=$(sbatch --requeue --parsable "$@" | tail -n1)
echo "	$PID"

