import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext

import numpy as np
from pypdf import PdfReader
from openai import OpenAI

# ==========================
# CONFIGURATION
# ==========================

API_KEY = "YOUR_OPENAI_API_KEY"

client = OpenAI(api_key=API_KEY)

knowledge_chunks = []
knowledge_embeddings = []

# ==========================
# PDF LOADING
# ==========================

def load_pdf(pdf_path):

    reader = PdfReader(pdf_path)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text

# ==========================
# TEXT CHUNKING
# ==========================

def split_text(text, chunk_size=1000, overlap=200):

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunks.append(text[start:end])

        start += chunk_size - overlap

    return chunks

# ==========================
# EMBEDDINGS
# ==========================

def create_embedding(text):

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )

    return response.data[0].embedding

# ==========================
# COSINE SIMILARITY
# ==========================

def cosine_similarity(v1, v2):

    v1 = np.array(v1)
    v2 = np.array(v2)

    return np.dot(v1, v2) / (
        np.linalg.norm(v1) * np.linalg.norm(v2)
    )

# ==========================
# BUILD KNOWLEDGE BASE
# ==========================

def build_knowledge_base(pdf_path):

    global knowledge_chunks
    global knowledge_embeddings

    knowledge_chunks.clear()
    knowledge_embeddings.clear()

    text = load_pdf(pdf_path)

    chunks = split_text(text)

    status_label.config(
        text=f"Processing {len(chunks)} chunks..."
    )

    root.update()

    for chunk in chunks:

        embedding = create_embedding(chunk)

        knowledge_chunks.append(chunk)
        knowledge_embeddings.append(embedding)

    status_label.config(
        text="Knowledge Base Ready"
    )

# ==========================
# RETRIEVE CONTEXT
# ==========================

def retrieve_context(question):

    question_embedding = create_embedding(question)

    scores = []

    for idx, emb in enumerate(
        knowledge_embeddings
    ):

        score = cosine_similarity(
            question_embedding,
            emb
        )

        scores.append((score, idx))

    scores.sort(reverse=True)

    context = ""

    for score, idx in scores[:3]:
        context += knowledge_chunks[idx]
        context += "\n\n"

    return context

# ==========================
# ASK QUESTION
# ==========================

def ask_question():

    question = question_entry.get()

    if not question:
        return

    context = retrieve_context(question)

    prompt = f"""
You are an educational tutor.

Answer only using the context.

Context:
{context}

Question:
{question}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2
    )

    answer = response.choices[0].message.content

    answer_box.delete("1.0", tk.END)
    answer_box.insert(tk.END, answer)

# ==========================
# PDF UPLOAD
# ==========================

def upload_pdf():

    pdf_path = filedialog.askopenfilename(
        filetypes=[("PDF Files", "*.pdf")]
    )

    if pdf_path:

        build_knowledge_base(pdf_path)

        messagebox.showinfo(
            "Success",
            "PDF Loaded Successfully"
        )

# ==========================
# GUI
# ==========================

root = tk.Tk()

root.title("Educational RAG Tutor")

root.geometry("800x600")

title = tk.Label(
    root,
    text="Educational RAG Tutor",
    font=("Arial", 18, "bold")
)

title.pack(pady=10)

upload_btn = tk.Button(
    root,
    text="Upload PDF",
    command=upload_pdf
)

upload_btn.pack()

status_label = tk.Label(
    root,
    text="No PDF Loaded"
)

status_label.pack(pady=5)

question_entry = tk.Entry(
    root,
    width=80
)

question_entry.pack(pady=10)

ask_btn = tk.Button(
    root,
    text="Ask Question",
    command=ask_question
)

ask_btn.pack()

answer_box = scrolledtext.ScrolledText(
    root,
    width=90,
    height=20
)

answer_box.pack(pady=10)

root.mainloop()
