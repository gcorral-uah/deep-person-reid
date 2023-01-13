#!/usr/bin/env bash
#The OOM Killer sends SIGKILL, so there's no way to gracefully handle the
#killing. But we can simply orphan the process to avoid the surrounding loop
#being terminated: See
#https://superuser.com/questions/948718/restart-a-command-after-it-gets-killed-by-the-system
# Another option may be
# https://askubuntu.com/questions/832767/run-another-command-when-my-script-receives-sigkill-signal
# Yeet another option may be to simply start the script without nohup, which
# seems to cause problems. In testing the OS doesn't kill the parent when a
# child recives a SIGKILL or SIGABORT (kill -9/-15).
while true; do
    if [[ -e "finished.txt" ]]; then
        break
    fi
    # conda run -n imagenet-training nohup python imagenet.py
    # nohup conda run -n imagenet-training python imagenet.py
    # source ~/miniconda3/etc/profile.d/conda.sh
    # conda activate imagenet-training
    # python imagenet.py 2> "nohup.err"
    conda run -n imagenet-training python imagenet.py 
done
