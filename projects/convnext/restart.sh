#!/usr/bin/env bash
#The OOM Killer sends SIGKILL, so there's no way to gracefully handle the
#killing. But we can simply orphan the process to avoid the surrounding loop
#being terminated: See
#https://superuser.com/questions/948718/restart-a-command-after-it-gets-killed-by-the-system
while true; do
    if [[ -e "finished.txt" ]]; then
        break
    fi
    conda run -n imagenet-training nohup python imagenet.py
    pid=$!
    wait $pid || continue
    break
done
