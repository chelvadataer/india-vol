#!/usr/bin/env bash
# Run src/chronos_job.py in the yug-base container with the Yug episode's Chronos venv (read-only, offline HF cache).
# GPU when available (Shorts have GPU priority: keep runs short, or set CHRONOS_GPU=0 for CPU).
EP=/home/chelvas/server_setup/film-stack/projects/rom/ep10_yug
GPU=(); [ "${CHRONOS_GPU:-1}" = "1" ] && GPU=(--gpus all)
exec docker run --rm "${GPU[@]}" -u "$(id -u):$(id -g)" -e HOME=/ep/home -e HF_HOME=/ep/home/hf -e HF_HUB_OFFLINE=1 \
  -v "$EP":/ep:ro -v /home/chelvas/india-vol-data:/data -v /home/chelvas/server_setup/india-vol/src:/src:ro \
  -e FT_BATCH="${FT_BATCH:-64}" --memory 12g yug-base:local /ep/.venv_chronos/bin/python /src/chronos_job.py "$@"
