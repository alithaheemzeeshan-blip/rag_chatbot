import streamlit as st
import pdfplumber
from openai import OpenAI
import os
import json

# -------------------- BASIC SETTINGS --------------------
st.set_page_config(page_title="Zeeshan ka Chatbot", layout="centered")
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

PDF_PATH = "data/Zeeshan_Chatbot_Company_Manual.pdf"
EMBED_CACHE = "data/embeddings_cache.json"

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


# -------------------- SPLIT PDF INTO CHUNKS --------------------
def split_into_chunks(text, chunk_size=500):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks


# -------------------- EMBEDDINGS (ONLY ONCE!) --------------------
def embed_once_and_save():
    if os.path.exists(EMBED_CACHE):
        with open(EMBED_CACHE, "r") as f:
            return json.load(f)

    chunks = split_into_chunks(pdf_text)
    cache = []

    for chunk in chunks:
        emb = client.embeddings.create(
            model="text-embedding-3-small",
            input=chunk
        ).data[0].embedding

        cache.append({"text": chunk, "embedding": emb})

    with open(EMBED_CACHE, "w") as f:
        json.dump(cache, f)

    return cache


embeddings_cache = embed_once_and_save()


# -------------------- FIND MOST RELEVANT CHUNK --------------------
import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def retrieve_relevant_chunk(question):
    q_emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=question
    ).data[0].embedding

    best = None
    best_score = -1

    for item in embeddings_cache:
        score = cosine_similarity(np.array(q_emb), np.array(item["embedding"]))
        if score > best_score:
            best = item["text"]
            best_score = score

    return best


# -------------------- GET AI ANSWER --------------------
def get_answer(question):

    relevant_text = retrieve_relevant_chunk(question)

    system_prompt = f"""
You are Zeeshan ka Chatbot. Use the RAG system:
1. First use the PDF context **IF it is relevant**.
2. If the PDF does not contain the answer, use your own updated AI knowledge.

PDF CONTEXT:
{relevant_text}
"""

    res = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]
    )

    return res.choices[0].message["content"]


# -------------------- UI --------------------
st.title("🤖 Zeeshan ka Chatbot")

if "chat" not in st.session_state:
    st.session_state.chat = []

with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("Ask something:")
    submitted = st.form_submit_button("Send")

if submitted and user_input.strip():
    answer = get_answer(user_input)
    st.session_state.chat.append(("You", user_input))
    st.session_state.chat.append(("Bot", answer))

st.write("---")
for sender, msg in st.session_state.chat:
    if sender == "You":
        st.markdown(f"**🧑 You:** {msg}")
    else:
        st.markdown(f"**🤖 Zeeshan ka Chatbot:** {msg}")
