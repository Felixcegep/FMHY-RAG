# 🧠 FMHY RAG Assistant

Un assistant **RAG** (Retrieval-Augmented Generation) local basé sur [Ollama](https://ollama.com/) et **FAISS**, conçu pour répondre à des questions à partir de contenus Markdown extraits du site [https://fmhy.net/](https://fmhy.net/).

---

## ✅ Prérequis

### 🧩️ Dépendances système

- **Python 3.9+**
- **[Ollama](https://ollama.com/download)** installé et fonctionnel
- **jq**, **curl**, **wget** (inclus sur la plupart des distributions Linux/macOS)

### 📦 Modules Python

Installe les dépendances Python avec :

```bash
pip install faiss-cpu numpy tqdm flask
```

---

## 📅 Téléchargement des modèles Ollama

Avant toute exécution, télécharge les modèles nécessaires :

```bash
ollama pull nomic-embed-text
ollama run artifish/llama3.2-uncensored
```

> 🚪 Ouvre **Ollama** lorsque tu exécutes le programme.

---

## ⚙️ Étapes d'installation et d'exécution

1. **Clone le dépôt :**

```bash
git clone https://github.com/Felixcegep/FMHY-RAG.git
cd FMHY-RAG
```

2. **Télécharge les fichiers Markdown :**

- *Linux/macOS :*
```bash
bash setup/download_docs.sh
```

- *Windows (PowerShell) :*
```powershell
.\setup\download_docs.ps1
```

3. **Découpe les documents en sections :**

- *Linux/macOS :*
```bash
bash setup/split_all_docs.sh
```

- *Windows (PowerShell) :*
```powershell
.\setup\split_all_docs.ps1
```

4. **Construit l’index FAISS :**

```bash
python build_index.py
```

5. **Lance l’application web :**

```bash
python app.py
```

Puis ouvre ton navigateur sur 👉 [http://localhost:5000](http://localhost:5000) pour discuter avec le chatbot.

---

## 📁 Arborescence du projet

```
.
├── app.py                 # Application Flask pour discuter avec le chatbot
├── ask.py                 # Script CLI pour poser des questions manuellement
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

## 💬 Exemple d’utilisation

```bash
$ python app.py
```

→ Puis pose ta question dans l’interface web, comme :  
**"Where can I watch Korean dramas?"**

---

## 🔗 Crédit

Basé sur les contenus du site : [https://fmhy.net/](https://fmhy.net/)
