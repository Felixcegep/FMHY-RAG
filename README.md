# 🧠 FMHY RAG Assistant

Un assistant RAG (Retrieval-Augmented Generation) local basé sur [Ollama](https://ollama.com/) et FAISS, conçu pour répondre à des questions sur des contenus Markdown extraits du site web [https://fmhy.net/](https://fmhy.net/).

---

## ✅ Prérequis

### 🧩 Dépendances système

* **Python 3.9+**
* **[Ollama](https://ollama.com/download)** installé et fonctionnel
* **jq**, **curl**, **wget** (inclus sur la plupart des distributions Linux/macOS)

### 📦 Modules Python

Installe les dépendances Python avec :

```bash
pip install faiss-cpu numpy tqdm
```

---

## 📅 Téléchargement des modèles Ollama

Avant toute exécution, télécharge les modèles nécessaires :

```bash
ollama pull nomic-embed-text
ollama run artifish/llama3.2-uncensored
```

> 🚪 Ouvre Ollama quand tu veux executer le program

---

## ⚙️ Étapes d'installation et d'exécution

1. **Clone le dépôt** :

   ```bash
   git clone https://github.com/<ton-user>/<ton-repo>.git
   cd <ton-repo>
   ```

2. **Télécharge les fichiers Markdown** :

   ```bash
   bash setup/download_docs.sh
   ```

3. **Découpe les documents en sections** :

   ```bash
   bash setup/split_all_docs.sh
   ```

4. **Construis l'index FAISS** :

   ```bash
   python build_index.py
   ```

5. **Pose ta question** :

   ```bash
   python ask.py "Show me where I can watch Korean dramas."
   ```

---

## 📁 Arborescence du projet

```
.
├── ask.py               # Script principal pour poser des questions
├── build_index.py       # Indexation vectorielle des chunks
├── index.faiss          # Fichier d'index FAISS
├── passages.json        # Fichier contenant tous les chunks indexés
├── sections/            # Fichiers Markdown découpés par sections
├── setup/
│   ├── download_docs.sh # Script de téléchargement .md
│   └── split_all_docs.sh# Script de découpe en sections
└── docs/                # Fichiers Markdown d'origine
```

---

## 💬 Exemple d'utilisation

```bash
$ python ask.py "What are the best sites to download audiobooks?"
😎 Loaded index with 2945 passages
🔍 Searching for: What are the best sites to download audiobooks?
📚 Found 6 relevant passages from 4 sources:
  • Audiobooks_1.md
  • Audiobooks_2.md
...
```
