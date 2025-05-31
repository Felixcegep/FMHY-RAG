#!/bin/bash

mkdir -p docs

curl -s https://api.github.com/repos/fmhy/edit/contents/docs \
| jq -r '.[] | select(.name | endswith(".md")) | .download_url' \
| while read url; do
  filename=$(basename "$url")
  wget -q -O "docs/$filename" "$url" && echo "✅ Downloaded $filename"
done

