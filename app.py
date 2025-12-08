import os
import numpy as np
import streamlit as st
import pdfplumber
from openai import OpenAI

# -----------------------------
# 🔑 OpenAI client (uses Streamlit Secrets)
# -----------------------------
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# -----------------------------
# 📄 PDF loading & chunking
# -----------------------------
PDF_FILENAME = "Zeeshan_Chatbot_Company_Manual.pdf"

def find_pdf_path():
    # Try root and /data
    candidates = [
        PDF_FILENAME,
        os.path.join("data", PDF_FILENAME),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None

def load_pdf_chunks(chunk_chars: int = 800):
    pdf_path = find_pdf_path()
    if not pdf_path:
        return ["(No PDF file found. RAG will only use general AI knowledge.)"]

    full_text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                txt = page.extract_text() or ""
                full_text += txt + "\n"
    except Exception as e:
        return [f"(Error reading PDF: {e})"]

    # Simple character-based chunking
    chunks = []
    current = ""
    for line in full_text.split("\n"):
        if len(current) + len(line) + 1 > chunk_chars:
            if current.strip():
                chunks.append(current.strip())
            current = line + "\n"
        else:
            current += line + "\n"
    if current.strip():
        chunks.append(current.strip())

    return chunks

# -----------------------------
# 🧠 Embeddings & similarity
# -----------------------------
def get_embedding(text: str):
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=[text]
    )
    return resp.data[0].embedding

def cosine_sim(a, b):
    a = np.array(a)
    b = np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

def build_knowledge_base():
    chunks = load_pdf_chunks()
    kb = []
    for c in chunks:
        emb = get_embedding(c)
        kb.append({"text": c, "embedding": emb})
    return kb

def retrieve_relevant_chunks(question: str, k: int = 4):
    # Build KB once and cache in session
    if "knowledge_base" not in st.session_state:
        with st.spinner("Indexing PDF for RAG (one-time)…"):
            st.session_state.knowledge_base = build_knowledge_base()

    kb = st.session_state.knowledge_base
    q_emb = get_embedding(question)

    scored = []
    for item in kb:
        score = cosine_sim(q_emb, item["embedding"])
        scored.append((score, item["text"]))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = [t for _, t in scored[:k]]
    return "\n\n---\n\n".join(top)

# -----------------------------
# 🤖 Answer generation with RAG
# -----------------------------
def generate_answer(question: str):
    context = retrieve_relevant_chunks(question)

    system_prompt = """
You are **Zeeshan ka Chatbot**, a corporate professional, techy AI assistant.
Answer clearly, politely, and in simple language. 
Prefer the provided context when relevant. If something is not in the context,
you may still answer using your general AI knowledge, but NEVER invent company policies.
"""

    user_prompt = f"""
CONTEXT FROM COMPANY PDF (may be partial):

{context}

---

USER QUESTION:
{question}

Using the context above first, answer the user's question.
If context is missing or incomplete, say so briefly and then answer with general knowledge if safe.
"""

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )

    return resp.choices[0].message.content

# -----------------------------
# 🎨 Streamlit UI setup
# -----------------------------
st.set_page_config(
    page_title="Zeeshan ka Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.markdown(
    """
    <style>
    .chat-bubble-user {
        background-color: #1e1e1e;
        padding: 10px 14px;
        border-radius: 12px;
        margin-bottom: 8px;
        color: #ffffff;
    }
    .chat-bubble-bot {
        background-color: #0b3d91;
        padding: 10px 14px;
        border-radius: 12px;
        margin-bottom: 15px;
        color: #ffffff;
    }
    .chat-container {
        max-height: 500px;
        overflow-y: auto;
        padding-right: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    "<h1 style='text-align:center;'>🤖 Zeeshan ka Chatbot</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center;'>Advanced RAG chatbot using your company PDF + OpenAI.</p>",
    unsafe_allow_html=True
)

st.divider()

# -----------------------------
# 💾 Session state
# -----------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "input_box" not in st.session_state:
    st.session_state.input_box = ""

# -----------------------------
# 🧑‍💻 Input area
# -----------------------------
user_input = st.text_input(
    "Ask something:",
    key="input_box",
    placeholder="Example: What are the safety policies?"
)

if st.button("Send"):
    if user_input.strip():
        answer = generate_answer(user_input)

        st.session_state.chat_history.append(
            {"role": "user", "content": user_input}
        )
        st.session_state.chat_history.append(
            {"role": "assistant", "content": answer}
        )

        # Clear box on next run
        st.session_state.input_box = ""

# -----------------------------
# 💬 Chat history display
# -----------------------------
st.markdown("### 💬 Conversation")
st.markdown("<div class='chat-container'>", unsafe_allow_html=True)

for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.markdown(
            f"<div class='chat-bubble-user'><b>🧑 You:</b><br>{msg['content']}</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<div class='chat-bubble-bot'><b>🤖 Zeeshan ka Chatbot:</b><br>{msg['content']}</div>",
            unsafe_allow_html=True
        )

st.markdown("</div>", unsafe_allow_html=True)
