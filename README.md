# IR Search Engine v2

A Flask web application implementing a Search Engine using Inverted Index + TF-IDF + Cosine Similarity — with support for **user-uploaded documents**.

## Features

- 🔍 Search across documents using TF-IDF + Cosine Similarity
- 📂 Upload your own `.txt` documents
- 🗑️ Delete uploaded documents
- 📄 5 default documents included
- 📊 Shows all calculation tables: Inverted Index, TF, IDF, TF-IDF, Query Vector

## Project Structure

```
search_engine_v2/
├── app.py
├── requirements.txt
├── README.md
├── default_docs/
│   ├── d1.txt
│   ├── d2.txt
│   ├── d3.txt
│   ├── d4.txt
│   └── d5.txt
├── uploads/          ← user uploaded files go here
└── templates/
    ├── index.html
    └── documents.html
```

## Installation

```bash
git clone https://github.com/<your-username>/search-engine-v2.git
cd search-engine-v2
pip install -r requirements.txt
python app.py
```

Open: **http://127.0.0.1:5000**

## How It Works

1. Load default + uploaded documents
2. Preprocess — lowercase, remove stopwords
3. Build Inverted Index
4. Calculate TF, IDF, TF-IDF
5. Convert query to TF-IDF vector
6. Rank documents by Cosine Similarity

## Deployment

Deployed on Railway:  https://web-production-aa438.up.railway.app/
