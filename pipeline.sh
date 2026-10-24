#!/usr/bin/env bash
# IndiaVol end-to-end run, exactly the frozen plan (docs/2026-10-24-design-and-prereg.md + addendum):
# wait for the 1m fetch -> panel -> export -> B2 zero-shot preds -> 3 M1 fine-tunes (CPU; GPU belongs to Shorts)
# -> M1 val preds -> evaluate val (FREEZES selection) -> holdout preds (B2 + selected M1) -> evaluate holdout (once).
set -euo pipefail
cd /home/chelvas/server_setup/india-vol
V=/home/chelvas/server_setup/upstox-trade-bot/.venv/bin/python
D=/home/chelvas/india-vol-data
export CHRONOS_GPU=0
log() { echo "=== $(date '+%F %T') $*"; }
# wait for the 1m fetch; if it died before finishing (e.g. its parent session exited), restart it — it resumes from
# the month cache — so training never starts on a partial universe
fetch_complete() {
  local total done
  # only the LAST run counts: a restarted fetch appends a new header and numbers from [1/N] again
  total=$(grep -oE '^[0-9]+ F&O underlyings' "$D/fetch.log" | tail -1 | cut -d' ' -f1)
  done=$(awk '/^[0-9]+ F&O underlyings/{n=0} /^\[[0-9]+\/[0-9]+\] /{n++} END{print n+0}' "$D/fetch.log")
  [ -n "$total" ] && [ "$done" -ge "$total" ]
}
while ! fetch_complete; do
  if ! pgrep -f "src/fetch_[1]m.py" >/dev/null; then
    log "fetch not running and not complete — restarting it (resumes from cache)"
    setsid nohup "$V" src/fetch_1m.py >> "$D/fetch.log" 2>&1 < /dev/null &
    sleep 120
  fi
  sleep 60
done
log "fetch done: $(grep -c ' bars$' $D/fetch.log) ok, $(grep -c FAILED $D/fetch.log) failed"
$V src/panel.py
$V src/export.py
log "B2 zero-shot predictions"
./run_chronos.sh predict zeroshot /data/origins_val.csv /data/pred_b2_zeroshot_val.csv
./run_chronos.sh predict zeroshot /data/origins_holdout.csv /data/pred_b2_zeroshot_holdout.csv
for spec in "m1_lora_1000 lora 1e-5 1000" "m1_lora_3000 lora 1e-5 3000" "m1_full_1000 full 1e-6 1000"; do
  set -- $spec
  log "fine-tune $1"
  ./run_chronos.sh finetune "$1" "$2" "$3" "$4"
  ./run_chronos.sh predict "/data/models/$1/final" /data/origins_val.csv "/data/pred_$1_val.csv"
done
log "validation + selection (frozen)"
$V src/evaluate.py val
M1=$($V -c "import json; print(json.load(open('$D/selection.json'))['m1'])")
log "holdout predictions for $M1"
./run_chronos.sh predict "/data/models/$M1/final" /data/origins_holdout.csv "/data/pred_${M1}_holdout.csv"
log "HOLDOUT (opened once)"
$V src/evaluate.py holdout
log "done"
