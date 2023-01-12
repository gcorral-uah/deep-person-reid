#!/usr/bin/env bash
while true; do
    if [[ -e "finished.txt" ]]
        break
    fi
    python imagenet.py
done
