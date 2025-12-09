import streamlit as st
import pdfplumber
import json
import os
from openai import OpenAI
import numpy as np

st.set_page_config(page_title="Zeeshan ka Chatbot", layout="centered")

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

PDF_PATH = "data/Zeeshan_Chatbot_Company_Manual.pdf"
EMBEDDINGS_FILE = "data/pdf_embeddings.json"

# -------------------- LOAD PDF --------------------
@st.cache_data
def load_pdf_text():
    text = ""
    with pdfplumber.open(PDF_PATH) as pdf:
        for page in pdf.pages:
            tx = page.extract_text()
            if tx:
                text += tx + "\n"
    return text

pdf_text = load_pdf_text()

# -------------------- SPLIT PDF --------------------
def chunk_text(text, size=500):
    words = text.split()
    return [" ".join(words[i:i+size]) for i in range(0, len(words), size)]

chunks = chunk_text(pdf_text)

# -------------------- CREATE EMBEDDINGS ONCE --------------------
def generate_embeddings_once():
    """Creates embeddings only ONCE and saves locally."""
    if os.path.exists(EMBEDDINGS_FILE):
        with open(EMBEDDINGS_FILE, "r") as f:
            return json.load(f)

    st.warning("📌 First-time setup: Creating embeddings...")

    embeddings = []
    for chunk in chunks:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=chunk
        )
        emb = response.data[0].embedding
        embeddings.append(emb)

    with open(EMBEDDINGS_FILE, "w") as f:
        json.dump(embeddings, f)

    return embeddings

embeddings = generate_embeddings_once()

# -------------------- FIND BEST MATCH --------------------
def search_pdf(query):
    """Find closest chunk using cosine similarity."""
    query_emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    ).data[0].embedding

    scores = []
    for i, emb in enumerate(embeddings):
        emb = np.array(emb)
        q = np.array(query_emb)
        score = np.dot(emb, q) / (np.linalg.norm(emb) * np.linalg.norm(q))
        scores.append((score, i))

    scores.sort(reverse=True)
    best_chunk = chunks[scores[0][1]]
    return best_chunk

# -------------------- AI ANSWERING --------------------
def get_answer(question):
    context = search_pdf(question)

    prompt = f"""
You are Zeeshan ka Chatbot.

Use the PDF information **if relevant**, BUT also use **your updated 2024 knowledge**.

PDF REFERENCE:
\"\"\"
{context}
\"\"\"
"""

    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": question}
        ]
    )
    return res.choices[0].message["content"]

# -------------------- UI --------------------
st.title("🤖 Zeeshan ka Chatbot (RAG + AI)")

if "chat" not in st.session_state:
    st.session_state.chat = []

with st.form("ask_form", clear_on_submit=True):
    question = st.text_input("Ask something:")
    submitted = st.form_submit_button("Send")

if submitted and question.strip():
    answer = get_answer(question)
    st.session_state.chat.append(("You", question))
    st.session_state.chat.append(("Bot", answer))

st.write("---")
for sender, msg in st.session_state.chat:
    if sender == "You":
        st.markdown(f"**🧑 You:** {msg}")
    else:
        st.markdown(f"**🤖 Zeeshan ka Chatbot:** {msg}")
