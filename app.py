import streamlit as st
import os
from openai import OpenAI
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Zeeshan ka Chatbot",
    page_icon="🤖",
    layout="centered",
)

# -----------------------------
# API Key Load
# -----------------------------
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
client = OpenAI()

system_prompt = """
You are Zeeshan ka Chatbot — a friendly, professional AI assistant.
Use PDF company knowledge + general AI to answer accurately.
"""

# -----------------------------
# Load PDF + Build Vector DB
# -----------------------------
@st.cache_resource
def load_kb():
    pdf_path = "data/Zeeshan_Chatbot_Company_Manual.pdf"

    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"

    chunks = []
    size = 500
    for i in range(0, len(text), size):
        chunks.append(text[i:i + size])

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))

    return chunks, embeddings, index, model


chunks, embeddings, index, model = load_kb()

# -----------------------------
# PDF Search
# -----------------------------
def search_docs(query):
    q_emb = model.encode([query])
    k = 3
    dist, ids = index.search(np.array(q_emb), k)
    return "\n\n".join([chunks[i] for i in ids[0]])

# -----------------------------
# GPT Answer
# -----------------------------
def generate_answer(question):
    context = search_docs(question)

    prompt = f"""
User question: {question}

Relevant company knowledge:
{context}

Now reply professionally as Zeeshan ka Chatbot.
"""

    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
    )
    return res.choices[0].message.content


# --------------------------------
# UI START
# --------------------------------
st.markdown(
    """
    <h1 style='text-align:center;color:#00E0FF;'>🤖 Zeeshan ka Chatbot</h1>
    """,
    unsafe_allow_html=True,
)

# Chat messages stored
if "chat" not in st.session_state:
    st.session_state.chat = []

# -----------------------------
# User input box (clears after question)
# -----------------------------
user_input = st.text_input("Ask something:", key="input_box")

send = st.button("Send")

if send and user_input.strip() != "":
    # store user message
    st.session_state.chat.append({"role": "user", "content": user_input})

    # generate reply
    with st.spinner("🤖 Thinking..."):
        answer = generate_answer(user_input)

    # store assistant reply
    st.session_state.chat.append({"role": "assistant", "content": answer})

    # CLEAR the input box
    st.session_state.input_box = ""

    # Refresh page AFTER clearing input
    st.rerun()


# -----------------------------
# SHOW only the LAST exchange under the input
# -----------------------------
if st.session_state.chat:
    last_msg = st.session_state.chat[-1]
    if last_msg["role"] == "assistant":
        st.markdown(
            f"""
            <div style='margin-top:20px; color:#00FFAA; font-size:18px;'>
            <b>🤖 Zeeshan ka Chatbot:</b><br>{last_msg["content"]}
            </div>
            """,
            unsafe_allow_html=True,
        )
