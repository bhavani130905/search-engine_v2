"""
Search Engine v2 — Inverted Index + TF-IDF + Cosine Similarity
Supports user-uploaded .txt documents + default document collection
Flask Web Application
"""

import math
import os
import shutil
from collections import defaultdict, Counter
from flask import Flask, render_template, request, redirect, url_for, flash

app = Flask(__name__)
app.secret_key = "searchengine_secret_key"

# ---------------------------
# DIRECTORIES
# ---------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DIR = os.path.join(BASE_DIR, "default_docs")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")

os.makedirs(DEFAULT_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ---------------------------
# STOPWORDS
# ---------------------------
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "is", "it", "its", "as", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "do",
    "does", "did", "will", "would", "could", "should", "may", "might",
    "this", "that", "these", "those", "they", "them", "their", "there",
    "we", "our", "you", "your", "he", "she", "his", "her", "i", "my",
    "not", "no", "so", "than", "then", "too", "very", "just", "also",
    "about", "into", "more", "even", "both", "each", "such", "how",
    "who", "what", "which", "any", "all", "one", "can", "now", "how"
}

# ---------------------------
# PREPROCESSING
# ---------------------------
def preprocess(text):
    tokens = text.lower().split()
    return [w.strip(".,!?;:()") for w in tokens if w.strip(".,!?;:()") and w.strip(".,!?;:()") not in STOPWORDS]

# ---------------------------
# LOAD ALL DOCUMENTS
# ---------------------------
def load_documents():
    docs, ids, sources = [], [], []

    # Default docs
    for f in sorted(os.listdir(DEFAULT_DIR)):
        if f.endswith(".txt"):
            with open(os.path.join(DEFAULT_DIR, f), "r", encoding="utf-8") as fp:
                docs.append(fp.read().strip())
            ids.append(os.path.splitext(f)[0].upper())
            sources.append("default")

    # Uploaded docs
    for f in sorted(os.listdir(UPLOAD_DIR)):
        if f.endswith(".txt"):
            with open(os.path.join(UPLOAD_DIR, f), "r", encoding="utf-8") as fp:
                docs.append(fp.read().strip())
            ids.append(os.path.splitext(f)[0].upper())
            sources.append("uploaded")

    return docs, ids, sources

# ---------------------------
# BUILD IR MODEL
# ---------------------------
def build_model(documents, doc_ids):
    tokenized = [preprocess(doc) for doc in documents]
    vocab = sorted(set(w for doc in tokenized for w in doc))
    N = len(documents)

    # Inverted index
    inv_index = defaultdict(list)
    for i, doc in enumerate(tokenized):
        for w in set(doc):
            inv_index[w].append(doc_ids[i])

    # IDF
    idf = {}
    df_counts = {}
    for term in vocab:
        df = sum(1 for doc in tokenized if term in doc)
        df_counts[term] = df
        idf[term] = round(math.log10(N / df), 4) if df > 0 else 0

    # TF-IDF vectors
    tfidf_vecs = []
    for doc in tokenized:
        c = Counter(doc)
        tfidf_vecs.append([round(c.get(t, 0) * idf[t], 4) for t in vocab])

    return tokenized, vocab, inv_index, idf, df_counts, tfidf_vecs

# ---------------------------
# COSINE SIMILARITY
# ---------------------------
def cosine_similarity(v1, v2):
    dot = sum(a * b for a, b in zip(v1, v2))
    n1 = math.sqrt(sum(a * a for a in v1))
    n2 = math.sqrt(sum(b * b for b in v2))
    return dot / (n1 * n2) if n1 and n2 else 0

# ---------------------------
# ROUTES
# ---------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    documents, doc_ids, sources = load_documents()
    results = None
    query_input = ""
    query_terms = []
    tables = {}

    if not documents:
        flash("No documents found. Please upload at least one .txt file.", "warning")
        return render_template("index.html", results=None, query_input="", query_terms=[],
                               tables={}, doc_count=0, doc_ids=[], sources=[])

    tokenized, vocab, inv_index, idf, df_counts, tfidf_vecs = build_model(documents, doc_ids)

    if request.method == "POST":
        query_input = request.form.get("query", "").strip()
        query_terms = preprocess(query_input)

        qc = Counter(query_terms)
        query_vec = [round(qc.get(t, 0) * idf.get(t, 0), 4) for t in vocab]

        sims = [round(cosine_similarity(query_vec, row), 4) for row in tfidf_vecs]
        ranked = sorted(zip(doc_ids, sims, sources), key=lambda x: x[1], reverse=True)

        best_score = ranked[0][1]
        top_docs = [d for d, s, _ in ranked if s == best_score and s > 0]

        if best_score == 0:
            result_msg = f"No relevant documents found for '{query_input}'."
        elif len(top_docs) == 1:
            result_msg = f"'{query_input}' is most relevant to {top_docs[0]} with a cosine similarity of {best_score}."
        else:
            result_msg = f"'{query_input}' is equally relevant to {', '.join(top_docs)} with a cosine similarity of {best_score}."

        results = {"ranked": ranked, "result_msg": result_msg, "best_score": best_score, "top_docs": top_docs}

        # Tables for query terms only
        display_terms = [t for t in vocab if t in query_terms]
        if display_terms:
            tf_data = []
            for i, doc in enumerate(tokenized):
                c = Counter(doc)
                tf_data.append([(t, c.get(t, 0)) for t in display_terms])

            tables = {
                "display_terms": display_terms,
                "inv_index": [(t, ", ".join(inv_index[t])) for t in display_terms],
                "tf": list(zip(doc_ids, tf_data)),
                "idf": [(t, df_counts[t], idf[t]) for t in display_terms],
                "tfidf": [(doc_ids[i], [(t, round(Counter(tokenized[i]).get(t, 0) * idf[t], 4)) for t in display_terms]) for i in range(len(doc_ids))],
                "query_vec": [(t, round(qc.get(t, 0) * idf.get(t, 0), 4)) for t in display_terms],
            }

    return render_template("index.html",
                           results=results,
                           query_input=query_input,
                           query_terms=query_terms,
                           tables=tables,
                           doc_count=len(documents),
                           doc_ids=doc_ids,
                           sources=sources,
                           vocab_size=len(vocab) if documents else 0)


@app.route("/upload", methods=["POST"])
def upload():
    files = request.files.getlist("files")
    uploaded = 0
    for file in files:
        if file and file.filename.endswith(".txt"):
            safe_name = file.filename.replace(" ", "_")
            file.save(os.path.join(UPLOAD_DIR, safe_name))
            uploaded += 1
    if uploaded:
        flash(f"{uploaded} file(s) uploaded successfully!", "success")
    else:
        flash("Please upload valid .txt files only.", "error")
    return redirect(url_for("documents"))


@app.route("/delete/<doc_id>")
def delete(doc_id):
    filename = doc_id.lower() + ".txt"
    path = os.path.join(UPLOAD_DIR, filename)
    if os.path.exists(path):
        os.remove(path)
        flash(f"{doc_id} deleted successfully.", "success")
    else:
        flash(f"Could not find {doc_id} to delete.", "error")
    return redirect(url_for("documents"))


@app.route("/documents")
def documents():
    docs, ids, sources = load_documents()
    doc_list = []
    for doc_id, doc_text, source in zip(ids, docs, sources):
        lines = [l for l in doc_text.split('\n') if l.strip()]
        doc_list.append({"id": doc_id, "text": doc_text, "preview": lines[:2], "source": source})
    return render_template("documents.html", doc_list=doc_list)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
# v2 - added upload and delete functionality