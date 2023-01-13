#!/usr/bin/env bash
while true; do
    if [[ -e "finished.txt" ]]; then
        break
    fi
    python imagenet.py
done
