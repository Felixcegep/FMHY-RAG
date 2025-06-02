[media pointer="file-service://file-8vSwv3cEB8w6zGMxNYoQ78"]
so change the README TO INCLUDE THE LOCAL AND GOOGLE VERSIONS # 🧠 FMHY RAG Assistant

Un assistant **RAG** (Retrieval-Augmented Generation) local basé sur [Ollama](https://ollama.com/) et **FAISS**, conçu pour répondre à des questions à partir de contenus Markdown extraits du site [https://fmhy.net/](https://fmhy.net/).

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

## 🗕️ Téléchargement des modèles Ollama

Avant toute exécution, télécharge les modèles nécessaires :

```bash
ollama pull nomic-embed-text
ollama run artifish/llama3.2-uncensored
```

> 🚪 Ouvre Ollama lorsque tu exécutes le programme.

---

## ⚙️ Étapes d'installation et d'exécution

1. Clone le dépôt :

```bash
git clone https://github.com/Felixcegep/FMHY-RAG.git
cd FMHY-RAG
```

2. Télécharge les fichiers Markdown :

* **Linux/macOS** :

```bash
bash setup/download_docs.sh
```

* **Windows (PowerShell)** :

```powershell
.\setup\download_docs.ps1
```

3. Découpe les documents en sections :

* **Linux/macOS** :

```bash
bash setup/split_all_docs.sh
```

* **Windows (PowerShell)** :

```powershell
.\setup\split_all_docs.ps1
```

4. Construis l'index FAISS :

```bash
python build_index.py
```

5. Pose ta question :

```bash
python ask_local.py "Show me where I can watch Korean dramas."
```

---

## 📁 Arborescence du projet

```
.
├── ask.py                 # Script principal pour poser des questions
├── build_index.py         # Indexation vectorielle des chunks
├── index.faiss            # Fichier d'index FAISS
├── passages.json          # Fichier contenant tous les chunks indexés
├── sections/              # Fichiers Markdown découpés par sections
├── setup/
│   ├── download_docs.sh   # Script Bash de téléchargement
│   ├── download_docs.ps1  # Script PowerShell de téléchargement
│   ├── split_all_docs.sh  # Script Bash de découpe
│   └── split_all_docs.ps1 # Script PowerShell de découpe
└── docs/                  # Fichiers Markdown d'origine
```

---

## 💬 Exemple d'utilisation

```bash
$ python ask_local.py "What are the best sites to download audiobooks?"
✅ Loaded index with 2945 passages
🔍 Searching for: What are the best sites to download audiobooks?
📚 Found 6 relevant passages from 4 sources:
  • Audiobooks_1.md
  • Audiobooks_2.md
...
```

---
