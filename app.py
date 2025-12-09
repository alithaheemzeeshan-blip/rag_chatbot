import streamlit as st
import pdfplumber
import requests
import textwrap

# -------------------- BASIC SETUP --------------------
st.set_page_config(page_title="Zeeshan ka Chatbot", layout="centered")

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
PDF_PATH = "data/Zeeshan_Chatbot_Company_Manual.pdf"
MODEL_NAME = "llama-3.1-8b-instant"   # UPDATED WORKING MODEL


# -------------------- LOAD + CHUNK PDF --------------------
@st.cache_data
def load_chunks(max_chars: int = 600):
    text = ""
    with pdfplumber.open(PDF_PATH) as pdf:
        for page in pdf.pages:
            tx = page.extract_text()
            if tx:
                text += tx + "\n"

    raw_parts = [p.strip() for p in text.split("\n") if p.strip()]
    chunks = []
    buf = ""

    for part in raw_parts:
        if len(buf) + len(part) <= max_chars:
            buf += " " + part
        else:
            chunks.append(buf.strip())
            buf = part

    if buf:
        chunks.append(buf.strip())

    return chunks


pdf_chunks = load_chunks()


# -------------------- SIMPLE RETRIEVAL --------------------
def retrieve_context(query: str, top_k: int = 3):
    q_words = set(query.lower().split())
    scored = []

    for ch in pdf_chunks:
        ch_words = set(ch.lower().split())
        score = len(q_words & ch_words)
        if score > 0:
            scored.append((score, ch))

    if not scored:
        return "\n\n".join(pdf_chunks[:top_k])

    scored.sort(reverse=True, key=lambda x: x[0])
    return "\n\n".join([c for _, c in scored[:top_k]])


# -------------------- CALL GROQ API --------------------
def llama_chat(messages):
    url = "https://api.groq.com/openai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": 0.4,
    }

    response = requests.post(url, json=payload, headers=headers)
    result = response.json()

    try:
        return result["choices"][0]["message"]["content"]
    except:
        return "⚠️ Groq API Error:\n" + str(result)


# -------------------- RAG ANSWER --------------------
def get_answer(question: str, history):
    context = retrieve_context(question)

    system_prompt = f"""
You are **Zeeshan ka Chatbot**.
Use PDF context first.
If PDF has no answer, use AI knowledge.
If question is real-time based, warn that information may be outdated.

PDF CONTEXT:
---------------------
{context}
---------------------
"""

    messages = [{"role": "system", "content": system_prompt}]

    for m in history[-6:]:
        messages.append(m)

    messages.append({"role": "user", "content": question})

    return llama_chat(messages)


# -------------------- STREAMLIT UI --------------------
st.title("🤖 Zeeshan ka Chatbot – LLaMA Powered (Groq)")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant",
         "content": "Assalam o Alaikum! 👋 Main **Zeeshan ka Chatbot** hoon. "
                    "PDF + AI dono mix karta hoon. Kuch bhi pooch lo!"}
    ]

# Display messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
user_input = st.chat_input("Apna sawal likho...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Soch raha hoon..."):
            answer = get_answer(user_input, st.session_state.messages)
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
