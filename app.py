import streamlit as st
import pdfplumber
import os
import json
from openai import OpenAI
import numpy as np

# -------------------------------------------------
# BASIC SETUP
# -------------------------------------------------
st.set_page_config(page_title="Zeeshan ka Chatbot", layout="centered")
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

PDF_PATH = "data/Zeeshan_Chatbot_Company_Manual.pdf"
EMBED_FILE = "embeddings_cache.json"

# -------------------------------------------------
# LOAD PDF
# -------------------------------------------------
def load_pdf_text():
    text = ""
    with pdfplumber.open(PDF_PATH) as pdf:
        for p in pdf.pages:
            t = p.extract_text()
            if t:
                text += t + "\n"
    return text

pdf_text = load_pdf_text()

# -------------------------------------------------
# CHUNK PDF
# -------------------------------------------------
def chunk_text(text, size=500):
    words = text.split()
    return [" ".join(words[i:i+size]) for i in range(0, len(words), size)]

chunks = chunk_text(pdf_text)

# -------------------------------------------------
# EMBEDDINGS: CREATE ONCE AND SAVE
# -------------------------------------------------
@st.cache_resource
def get_embeddings():
    if os.path.exists(EMBED_FILE):
        with open(EMBED_FILE, "r") as f:
            return json.load(f)

    emb_list = []
    st.write("⚠️ First-time setup: Creating embeddings...")

    for chunk in chunks:
        embedding = client.embeddings.create(
            model="text-embedding-3-small",
            input=chunk
        ).data[0].embedding
        emb_list.append(embedding)

    with open(EMBED_FILE, "w") as f:
        json.dump(emb_list, f)

    st.success("Embeddings saved!")
    return emb_list

embeddings = get_embeddings()

# -------------------------------------------------
# SEARCH FUNCTION
# -------------------------------------------------
def search_context(query, top_k=3):
    q_emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    ).data[0].embedding

    scores = []
    for i, e in enumerate(embeddings):
        sim = np.dot(q_emb, e) / (np.linalg.norm(q_emb) * np.linalg.norm(e))
        scores.append((sim, chunks[i]))

    scores.sort(reverse=True)
    best = [s[1] for s in scores[:top_k]]
    return "\n\n".join(best)

# -------------------------------------------------
# AI ANSWER USING BOTH KNOWLEDGE + PDF
# -------------------------------------------------
def get_answer(question):
    context = search_context(question)

    system_message = f"""
You are Zeeshan ka Chatbot. 
Use BOTH:
1) The provided PDF context.
2) Your latest AI knowledge (current world info, updated 2025).
If PDF contradicts AI knowledge, respond using the MOST ACCURATE AND LATEST INFORMATION.

PDF CONTEXT:
{context}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": question}
        ]
    )

    return response.choices[0].message.content

# -------------------------------------------------
# UI
# -------------------------------------------------
st.title("🤖 Zeeshan ka Chatbot (Real RAG + Live AI)")

if "chat" not in st.session_state:
    st.session_state.chat = []

with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("Ask something:")
    submitted = st.form_submit_button("Send")

if submitted and user_input.strip():
    reply = get_answer(user_input)
    st.session_state.chat.append(("You", user_input))
    st.session_state.chat.append(("Bot", reply))

# DISPLAY CHAT
st.write("----")
for sender, msg in st.session_state.chat:
    if sender == "You":
        st.markdown(f"**🧑 You:** {msg}")
    else:
        st.markdown(f"**🤖 Zeeshan ka Chatbot:** {msg}")
