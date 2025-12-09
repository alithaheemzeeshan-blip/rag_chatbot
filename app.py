import streamlit as st
import pdfplumber
import os
import pickle
import numpy as np
from openai import OpenAI

st.set_page_config(page_title="Zeeshan ka Chatbot", layout="centered")

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

PDF_PATH = "data/Zeeshan_Chatbot_Company_Manual.pdf"
EMBED_CACHE = "embeddings_cache.pkl"


# ------------------------ LOAD PDF ------------------------
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


# ------------------------ CHUNK PDF ------------------------
def chunk_text(text, size=400):
    words = text.split()
    return [" ".join(words[i:i+size]) for i in range(0, len(words), size)]

chunks = chunk_text(pdf_text)


# ------------------------ CREATE OR LOAD EMBEDDINGS ------------------------
def get_embedding(text):
    res = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return np.array(res.data[0].embedding)


def load_or_create_embeddings():
    if os.path.exists(EMBED_CACHE):
        return pickle.load(open(EMBED_CACHE, "rb"))

    st.warning("⏳ First-time setup: Creating embeddings...")

    embs = [get_embedding(ch) for ch in chunks]

    pickle.dump(embs, open(EMBED_CACHE, "wb"))
    return embs


embeddings = load_or_create_embeddings()


# ------------------------ FIND BEST MATCH ------------------------
def find_best_context(query):
    q_emb = get_embedding(query)

    sims = [np.dot(q_emb, e) / (np.linalg.norm(q_emb) * np.linalg.norm(e))
            for e in embeddings]

    best_idx = int(np.argmax(sims))
    return chunks[best_idx]


# ------------------------ AI ANSWER ------------------------
def get_answer(question):
    context = find_best_context(question)

    system_prompt = f"""
You are Zeeshan ka Chatbot.

Use the PDF information ONLY when relevant.
If the PDF lacks the answer (like current PM), then use your own updated knowledge.

PDF CONTEXT:
{context}
"""

    res = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]
    )

    return res.choices[0].message["content"]


# ------------------------ UI SETUP ------------------------
st.title("🤖 Zeeshan ka Chatbot")

if "chat" not in st.session_state:
    st.session_state.chat = []


# ------------------------ INPUT FORM ------------------------
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("Ask something:")
    submitted = st.form_submit_button("Send")

if submitted and user_input.strip():
    ans = get_answer(user_input)
    st.session_state.chat.append(("You", user_input))
    st.session_state.chat.append(("Bot", ans))


# ------------------------ DISPLAY CHAT ------------------------
st.write("---")
for sender, msg in st.session_state.chat:
    if sender == "You":
        st.markdown(f"**🧑 You:** {msg}")
    else:
        st.markdown(f"**🤖 Zeeshan ka Chatbot:** {msg}")
