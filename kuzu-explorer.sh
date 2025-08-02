#!/bin/bash

docker run -p 8001:8000 \
    -v /Users/knutole/umnidev/podcast-muncher/data:/database \
    -e KUZU_FILE=podcast.kuzu \
    --rm kuzudb/explorer:latest