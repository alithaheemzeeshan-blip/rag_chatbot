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
client = OpenAI()  # NEW OpenAI syntax

# -----------------------------
# Chatbot Personality
# -----------------------------
system_prompt = """
You are **Zeeshan ka Chatbot** — a friendly, highly professional corporate AI assistant.
Your personality:
- Helpful, accurate, clear.
- Professional but warm.
- Always confident.
- Writes easy-to-read answers.
- Uses short paragraphs and bullet points when useful.

You also have RAG knowledge from the uploaded PDF, so integrate that into your answers when helpful.
"""

# -----------------------------
# Load PDF & Build Vector DB
# -----------------------------
@st.cache_resource
def load_knowledgebase():
    pdf_path = "data/Zeeshan_Chatbot_Company_Manual.pdf"

    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"

    # Chunking
    chunks = []
    chunk_size = 500
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i + chunk_size])

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))

    return chunks, embeddings, index, model


chunks, embeddings, index, model = load_knowledgebase()

# -----------------------------
# RAG Search
# -----------------------------
def search_docs(query):
    query_embedding = model.encode([query])
    k = 3
    distances, ids = index.search(np.array(query_embedding), k)
    retrieved = [chunks[i] for i in ids[0]]
    return "\n\n".join(retrieved)


# -----------------------------
# Generate Answer (NEW OpenAI API)
# -----------------------------
def generate_answer(question):
    retrieved_context = search_docs(question)

    prompt = f"""
User question: {question}

Relevant company knowledge:
{retrieved_context}

Now give the final answer as Zeeshan ka Chatbot.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
    )

    return response.choices[0].message.content


# -----------------------------
# Streamlit UI
# -----------------------------
st.markdown(
    """
    <h1 style='text-align:center; color:#00E0FF;'>🤖 Zeeshan ka Chatbot</h1>
    <p style='text-align:center; font-size:18px; color:#d0d0d0;'>
        Your professional AI assistant powered by RAG + GPT.
    </p>
    <br>
    """,
    unsafe_allow_html=True,
)

# Chat history session storage
if "chat" not in st.session_state:
    st.session_state.chat = []

# Display chat messages
for msg in st.session_state.chat:
    if msg["role"] == "user":
        st.markdown(
            f"<div style='text-align:right; color:#FFD700; font-size:18px;'>🧑‍💼 {msg['content']}</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<div style='text-align:left; color:#00FFAA; font-size:18px;'>🤖 {msg['content']}</div>",
            unsafe_allow_html=True,
        )


# -----------------------------
# User Input Box
# -----------------------------
user_input = st.text_input("Ask something:", "")

submit = st.button("Send")

if submit and user_input.strip() != "":
    st.session_state.chat.append({"role": "user", "content": user_input})

    with st.spinner("🤖 Thinking..."):
        bot_reply = generate_answer(user_input)

    st.session_state.chat.append({"role": "assistant", "content": bot_reply})

    st.rerun()
