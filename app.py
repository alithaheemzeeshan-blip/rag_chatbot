import streamlit as st
import os
import openai
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Zeeshan ka Chatbot",
    page_icon="🤖",
    layout="centered",
)

# -----------------------------
# Title With Animation
# -----------------------------
st.markdown("""
<style>
.big-title {
    font-size: 45px;
    font-weight: 700;
    text-align: center;
    color: #00c3ff;
    animation: glow 2s ease-in-out infinite alternate;
}
@keyframes glow {
    from {text-shadow: 0 0 10px #0ff;}
    to {text-shadow: 0 0 25px #00d9ff;}
}
.chat-bubble-user {
    background: #1f2937;
    padding: 12px 18px;
    border-radius: 12px;
    margin: 8px 0;
}
.chat-bubble-bot {
    background: #111827;
    border-left: 4px solid #00c3ff;
    padding: 12px 18px;
    border-radius: 12px;
    margin: 8px 0;
}
</style>
<div class='big-title'>🤖 Zeeshan ka Chatbot</div>
""", unsafe_allow_html=True)


# -----------------------------
# API Key
# -----------------------------
openai.api_key = st.secrets["OPENAI_API_KEY"]
from openai import OpenAI
client = OpenAI(api_key=openai.api_key)

# -----------------------------
# Load PDF Knowledge (RAG)
# -----------------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")

DOCUMENTS = []
EMBEDDINGS = []

def load_pdfs():
    global DOCUMENTS, EMBEDDINGS
    DOCUMENTS = []
    EMBEDDINGS = []

    pdf_dir = "data"
    if not os.path.exists(pdf_dir):
        return

    for file in os.listdir(pdf_dir):
        if file.endswith(".pdf"):
            reader = PdfReader(os.path.join(pdf_dir, file))
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"

            chunks = text.split("\n\n")
            for chunk in chunks:
                cleaned = chunk.strip()
                if len(cleaned) > 50:
                    DOCUMENTS.append(cleaned)

    if DOCUMENTS:
        EMBEDDINGS = embedder.encode(DOCUMENTS)

load_pdfs()


# -----------------------------
# RAG Search
# -----------------------------
def retrieve_context(query):
    if not DOCUMENTS:
        return ""

    q_emb = embedder.encode([query])[0]
    similarities = np.dot(EMBEDDINGS, q_emb)

    top_k = similarities.argsort()[-3:][::-1]
    best_chunks = [DOCUMENTS[i] for i in top_k]

    return "\n\n".join(best_chunks)


# -----------------------------
# Generate Answer (FIXED)
# -----------------------------
def generate_answer(user_question: str) -> str:
    context = retrieve_context(user_question)

    system_msg = (
        "You are **Zeeshan ka Chatbot**, "
        "a respectful, professional and friendly corporate assistant.\n\n"
        "Rules:\n"
        "1. If company PDF knowledge matches the question, ALWAYS use it.\n"
        "2. If not, answer using general knowledge but mention: "
        "'Yeh maloomat company document se directly nahi mili, yeh general guidance hai.'\n"
        "3. Keep answers short, clear, polite."
    )

    if context:
        system_msg += (
            "\n\n--- COMPANY PDF KNOWLEDGE ---\n"
            f"{context}\n"
            "------------------------------"
        )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_question},
        ],
    )

    # FIX → handle both new + old message formats
    choice = response.choices[0]
    msg = choice.message

    if isinstance(msg, dict):
        return msg.get("content", "").strip()
    else:
        return getattr(msg, "content", "").strip()


# -----------------------------
# Chat History
# -----------------------------
if "history" not in st.session_state:
    st.session_state.history = []


# -----------------------------
# Chat UI
# -----------------------------
st.write("---")
user_input = st.text_input("Ask something from Zeeshan ka Chatbot:", "")

if st.button("Send"):
    if user_input.strip() != "":
        st.session_state.history.append(("You", user_input))
        bot_reply = generate_answer(user_input)
        st.session_state.history.append(("Bot", bot_reply))
        st.rerun()


# -----------------------------
# Display Chat Bubbles
# -----------------------------
st.write("### Conversation:")

for sender, msg in st.session_state.history:
    if sender == "You":
        st.markdown(f"<div class='chat-bubble-user'><b>You:</b> {msg}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='chat-bubble-bot'><b>Zeeshan ka Chatbot:</b> {msg}</div>", unsafe_allow_html=True)

