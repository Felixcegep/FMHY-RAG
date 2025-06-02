#!/bin/bash

# Move to the directory of the script (i.e., update_script/)
cd "$(dirname "$0")"

# Make sure output goes to ../docs from update_script/
mkdir -p ../docs

# Download .md files into ../docs
curl -s https://api.github.com/repos/fmhy/edit/contents/docs \
| jq -r '.[] | select(.name | endswith(".md")) | .download_url' \
| while read url; do
  filename=$(basename "$url")
  wget -q -O "../docs/$filename" "$url" && echo "✅ Downloaded $filename"
done
