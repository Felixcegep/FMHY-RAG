import json
import os
import re
import time
from tqdm import tqdm
import faiss
import numpy as np
import requests
import hashlib
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=API_KEY)

import re
import hashlib


def couper_bien(markdown_str):
    top_level_matches = list(re.finditer(r'^# (.+?)$', markdown_str, re.MULTILINE))
    subsections = []

    for i, match in enumerate(top_level_matches):
        parent_title = match.group(1).strip()
        start = match.start()
        end = top_level_matches[i + 1].start() if i + 1 < len(top_level_matches) else len(markdown_str)
        full_section = markdown_str[start:end].strip()

        # Remove the main header line to get content
        content_start = full_section.find('\n') + 1
        section_content = full_section[content_start:] if content_start > 0 else ""

        # Find all ## subsections
        sub_matches = list(re.finditer(r'^## (.+?)$', section_content, re.MULTILINE))

        if sub_matches:
            # Get content before first ## subsection
            first_sub_start = sub_matches[0].start()
            main_content = section_content[:first_sub_start].strip()

            # Add main content if it exists and is substantial
            if main_content and len(main_content) > 10:  # Avoid adding just "***" or empty content
                subsections.append({
                    "title": parent_title,
                    "section": parent_title,  # Use parent title as section name
                    "text": main_content,
                    "hashtext": hashlib.sha256(main_content.encode()).hexdigest()
                })

            # Process each ## subsection
            for j, sub_match in enumerate(sub_matches):
                sub_title = sub_match.group(1).strip()
                sub_start = sub_match.end() + 1  # Skip the newline after the header
                sub_end = sub_matches[j + 1].start() if j + 1 < len(sub_matches) else len(section_content)
                text = section_content[sub_start:sub_end].strip()

                # Only add if there's substantial content
                if text and len(text) > 10:
                    subsections.append({
                        "title": parent_title,
                        "section": sub_title,
                        "text": text,
                        "hashtext": hashlib.sha256(text.encode()).hexdigest()
                    })
        else:
            # No ## subsections, treat entire content as one section
            if section_content and len(section_content) > 10:
                subsections.append({
                    "title": parent_title,
                    "section": parent_title,
                    "text": section_content,
                    "hashtext": hashlib.sha256(section_content.encode()).hexdigest()
                })

    return subsections


def embed_with_retry(client, text, max_retries=3, base_delay=1):
    """Embed text with exponential backoff retry logic."""
    for attempt in range(max_retries):
        try:
            result = client.models.embed_content(
                model="text-embedding-004",
                contents=text,
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_DOCUMENT"
                )
            )
            return result.embeddings[0].values
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    print(f"Rate limit hit, waiting {delay} seconds before retry...")
                    time.sleep(delay)
                else:
                    raise e
            else:
                raise e


if __name__ == '__main__':
    embeddings = []
    extracted_data = requests.get("https://api.fmhy.net/single-page")
    data_bien = couper_bien(extracted_data.text)

    # Rate limiting: Max 140 requests per minute (leaving some buffer)
    requests_per_minute = 140
    delay_between_requests = 60.0 / requests_per_minute

    for i, test in enumerate(tqdm(data_bien, desc="🔍 Embedding sections")):
        print(f"Processing {test['title']} {test['section']}", {test['text']})

        try:
            embedding = embed_with_retry(client, test["text"])
            embeddings.append(embedding)

            # Add delay to respect rate limits (except for the last item)
            if i < len(data_bien) - 1:
                time.sleep(delay_between_requests)

        except Exception as e:
            print(f"Failed to embed section '{test['section']}': {e}")
            # You might want to skip this section or handle the error differently
            continue

    if embeddings:  # Only proceed if we have embeddings
        vectors = np.array(embeddings, dtype='float32')
        faiss.normalize_L2(vectors)
        dim = vectors.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(vectors)
        faiss.write_index(index, "index.faiss")

        with open("metadata.json", "w", encoding="utf-8") as f:
            json.dump(data_bien[:len(embeddings)], f, indent=2, ensure_ascii=False)

        print(f"Successfully processed {len(embeddings)} sections")
    else:
        print("No embeddings were generated")