#!/usr/bin/env bash
#The OOM Killer sends SIGKILL, so there's no way to gracefully handle the
#killing. But we can simply orphan the process to avoid the surrounding loop
#being terminated: See
#https://superuser.com/questions/948718/restart-a-command-after-it-gets-killed-by-the-system
while true; do
    if [[ -e "finished.txt" ]]; then
        break
    fi
    if [[ -e "nohup.out" ]]; then
        rm "nohup.out"
    fi
    # conda run -n imagenet-training nohup python imagenet.py
    # nohup conda run -n imagenet-training python imagenet.py
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate imagenet-training
    nohup python imagenet.py 2> "nohup.err"
    pid=$!
    wait $pid || continue
    break
done
