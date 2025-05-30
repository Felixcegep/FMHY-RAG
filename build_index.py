import sys, json, pathlib, textwrap
import faiss, numpy as np, requests
from tqdm import tqdm
import ollama
DOC_DIR, IDX_OUT, TEXT_OUT = sys.argv[1:]
EMBED_SERVER = "http://192.168.2.241:11434"

def emb(txt):
    res = requests.post(
        f"{EMBED_SERVER}/api/embeddings",
        json={"model": "nomic-embed-text", "prompt": txt}
    )
    res.raise_for_status()
    return np.array(res.json()["embedding"], dtype='float32')

passages = []
vectors = []

# Récupère tous les fichiers Markdown
md_files = list(pathlib.Path(DOC_DIR).glob('*.md'))

# Compte total de chunks pour afficher une barre de progression plus fine
total_chunks = 0
for md in md_files:
    contents = md.read_text(encoding='utf-8', errors='ignore')
    total_chunks += len(textwrap.wrap(contents, width=512))

# Barre de progression
with tqdm(total=total_chunks, desc="Embedding Markdown Chunks", unit="chunk") as pbar:
    for md in md_files:
        contents = md.read_text(encoding='utf-8', errors='ignore')
        for chunk in textwrap.wrap(contents, width=512):
            pid = len(passages)
            passages.append({'id': pid, 'text': chunk, 'src': md.name})
            vectors.append(emb(chunk))
            pbar.update(1)

mat = np.vstack(vectors)
index = faiss.IndexFlatIP(mat.shape[1])
faiss.normalize_L2(mat)
index.add(mat)
faiss.write_index(index, IDX_OUT)
json.dump(passages, open(TEXT_OUT, 'w', encoding='utf-8'))

print(f'✅ Indexed {len(passages)} chunks → {IDX_OUT}')
