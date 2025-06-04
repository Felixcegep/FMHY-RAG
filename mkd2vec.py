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


def couper_bien(markdown_str):
    top_level_matches = list(re.finditer(r'^# (.+?)\n', markdown_str, re.MULTILINE))
    subsections = []

    for i, match in enumerate(top_level_matches):
        parent_title = match.group(1).strip()
        start = match.end()
        end = top_level_matches[i + 1].start() if i + 1 < len(top_level_matches) else len(markdown_str)
        section_block = markdown_str[start:end].strip()

        # Step 2: Find all ## inside this top-level section
        sub_matches = list(re.finditer(r'^## (.+?)\n', section_block, re.MULTILINE))

        for j, sub_match in enumerate(sub_matches):
            sub_title = sub_match.group(1).strip()
            sub_start = sub_match.end()
            sub_end = sub_matches[j + 1].start() if j + 1 < len(sub_matches) else len(section_block)
            text = section_block[sub_start:sub_end].strip()

            subsections.append({
                "title": parent_title,
                "section": sub_title,
                "text": text,
                "hashtext": hashlib.sha256(text.encode()).hexdigest()
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
        print(f"Processing {test['section']}, length: {len(test['text'])}, hash: {test['hashtext']}")

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