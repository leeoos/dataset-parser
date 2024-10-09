#! /bin/bash

while true; do
    case "$1" in
        -c) rm -r local_cache/; shift ;;
        -d) rm -r local_data/;  shift ;;
        *) break ;;
    esac
done