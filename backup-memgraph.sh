#!/bin/bash

# Build custom Alpine with rsync on-the-fly
docker build -t alpine-rsync - > /dev/null 2>&1 <<EOF
FROM alpine:latest
RUN apk add --no-cache rsync
CMD ["rsync"]
EOF

docker run -it \
    --name memgraph-tools \
    --rm  \
    -v podcast-muncher_memgraph_data:/source \
    -v memgraph-backup-podcast-muncher:/backup \
    alpine-rsync \
    rsync -ah --checksum --progress /source/ /backup/