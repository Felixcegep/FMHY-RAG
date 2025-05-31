#!/bin/bash

cd "$(dirname "$0")"

INPUT_DIR="../docs"
OUTPUT_DIR="../sections"
mkdir -p "$OUTPUT_DIR"

for file in "$INPUT_DIR"/*.md; do
    echo "🔍 Processing $file..."
    current_file=""
    while IFS= read -r line; do
        if [[ "$line" =~ ►[[:space:]]*(.+) ]]; then
            raw_title="${BASH_REMATCH[1]}"
            title=$(echo "$raw_title" | sed -E 's/\[([^\]]+)\]\([^)]+\)/\1/g')
            filename="$(echo "$title" | tr ' ' '_' | tr -cd '[:alnum:]_').md"
            current_file="$OUTPUT_DIR/$filename"
            echo "# $title" > "$current_file"
        elif [[ -n "$current_file" ]]; then
            echo "$line" >> "$current_file"
        fi
    done < "$file"
done

echo "✅ All sections saved in: $OUTPUT_DIR"
