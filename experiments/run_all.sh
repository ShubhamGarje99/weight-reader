#!/bin/zsh
# Runs every experiment one at a time under nice, each with a hard timeout.
# Usage: zsh experiments/run_all.sh [epochs]
set -u
cd "$(dirname "$0")/.."
EP=${1:-60}
PY=.venv/bin/python
run() {  # name, args...
  local name=$1; shift
  if [[ -f runs.noindex/$name.json ]]; then echo "skip $name (done)"; return; fi
  echo "=== $name ($(date +%H:%M:%S))"
  nice -n 15 perl -e 'alarm 3600; exec @ARGV' $PY reader/train.py --epochs $EP --lr 2e-3 --threads 2 --name $name "$@" 2>&1 | grep --line-buffered -E "^encoder|^ep|Traceback|Error"
}
# (a) held-out seeds + (d) backdoor + (e) ablation
run gmn_seed            --encoder gmn  --split seed --save-emb
run flat_seed           --encoder flat --split seed
run flat_seed_permtest  --encoder flat --split seed --permute-test --eval-only flat_seed
run gmn_seed_permtest   --encoder gmn  --split seed --permute-test --eval-only gmn_seed
run meta_seed           --encoder meta --split seed
# (b) held-out widths
run gmn_width16         --encoder gmn  --split width --held 16
run gmn_width32         --encoder gmn  --split width --held 32
run flat_width16        --encoder flat --split width --held 16
# (c) held-out task family (family-agnostic heads only are meaningful here)
for fam in mnist geom2d modular boolean vector; do
  run gmn_fam_$fam      --encoder gmn  --split family --held $fam
done
echo "all done $(date +%H:%M:%S)"
