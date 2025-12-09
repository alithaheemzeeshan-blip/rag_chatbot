import streamlit as st
import pdfplumber
import json
import os
import numpy as np
from groq import Groq
from tavily import TavilyClient

# -----------------------------------------------------------
# SETUP
# -----------------------------------------------------------
st.set_page_config(page_title="Zeeshan RAG Chatbot", layout="centered")

GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "YOUR_GROQ_API_KEY")
TAVILY_API_KEY = st.secrets.get("TAVILY_API_KEY", "YOUR_TAVILY_API_KEY")

client = Groq(api_key=GROQ_API_KEY)
tavily = TavilyClient(api_key=TAVILY_API_KEY)

PDF_PATH = "data/Zeeshan_Chatbot_Company_Manual.pdf"
EMBED_FILE = "embeddings.json"


# -----------------------------------------------------------
# LOAD PDF TEXT
# -----------------------------------------------------------
@st.cache_data
def load_pdf():
    text = ""
    with pdfplumber.open(PDF_PATH) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
    return text


pdf_text = load_pdf()


# -----------------------------------------------------------
# LOCAL EMBEDDINGS (AVOID FAILING API CALLS)
# -----------------------------------------------------------
def simple_local_embedding(text):
    """Cheap, offline embedding (no API, no cost)."""
    words = text.lower().split()
    vec = np.zeros(300)

    for w in words:
        vec[hash(w) % 300] += 1

    return vec.tolist()


# -----------------------------------------------------------
# BUILD / LOAD EMBEDDINGS
# -----------------------------------------------------------
def build_or_load_embeddings():
    if os.path.exists(EMBED_FILE):
        return json.load(open(EMBED_FILE, "r"))

    chunks = [c for c in pdf_text.split("\n") if c.strip()]
    data = [{"text": c, "embedding": simple_local_embedding(c)} for c in chunks]

    json.dump(data, open(EMBED_FILE, "w"))
    return data


embeddings = build_or_load_embeddings()


# -----------------------------------------------------------
# RAG: FIND BEST PDF MATCH
# -----------------------------------------------------------
def retrieve_from_pdf(query):
    q_emb = simple_local_embedding(query)

    best_score = -1
    best_text = "No relevant PDF content found."

    for entry in embeddings:
        score = np.dot(q_emb, entry["embedding"])
        if score > best_score:
            best_score = score
            best_text = entry["text"]

    return best_text


# -----------------------------------------------------------
# TAVILY INTERNET SEARCH
# -----------------------------------------------------------
def internet_search(query):
    try:
        result = tavily.search(query=query, max_results=3)
        return result.get("results", [])
    except Exception:
        return []


# -----------------------------------------------------------
# GENERATE ANSWER USING BOTH RAG + TAVILY + GROQ
# -----------------------------------------------------------
def generate_answer(user_q):

    # 1️⃣ RAG retrieval
    pdf_context = retrieve_from_pdf(user_q)

    # 2️⃣ Live internet data
    search_results = internet_search(user_q)
    search_text = "\n".join([r["content"] for r in search_results]) if search_results else "No live data."

    # 3️⃣ Combine everything for Groq
    system_prompt = f"""
You are a hybrid RAG chatbot that uses PDF context AND real-time internet data.

PDF CONTEXT:
{pdf_context}

LIVE INTERNET DATA:
{search_text}

Use both sources when answering. 
If information conflicts, prefer the *latest internet data*.
"""

    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_q}
        ]
    )

    return response.choices[0].message["content"]


# -----------------------------------------------------------
# UI
# -----------------------------------------------------------
st.title("🤖 Zeeshan's Advanced RAG Chatbot (PDF + Internet)")

if "chat" not in st.session_state:
    st.session_state.chat = []

user_input = st.text_input("Ask anything:")

if st.button("Send") and user_input.strip():
    answer = generate_answer(user_input)
    st.session_state.chat.append(("You", user_input))
    st.session_state.chat.append(("Bot", answer))

st.write("---")
for sender, msg in st.session_state.chat:
    st.markdown(f"**{sender}:** {msg}")
