import streamlit as st
import pdfplumber
import numpy as np
from openai import OpenAI, RateLimitError

# -------------------- BASIC SETUP --------------------
st.set_page_config(page_title="Zeeshan ka Chatbot (RAG)", layout="wide")

# OpenAI client (key from Streamlit secrets)
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

PDF_PATH = "data/Zeeshan_Chatbot_Company_Manual.pdf"
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"


# -------------------- LOAD PDF & CHUNK IT --------------------
@st.cache_data(show_spinner="📄 Loading company manual...")
def load_pdf_text(path: str) -> str:
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


def chunk_text(text: str, chunk_chars: int = 900, overlap: int = 150):
    """Simple character-based chunking with overlap."""
    chunks = []
    start = 0
    length = len(text)

    while start < length:
        end = min(start + chunk_chars, length)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_chars - overlap

    return chunks


# -------------------- BUILD EMBEDDING INDEX (RAG) --------------------
@st.cache_data(show_spinner="🧠 Creating embeddings from your PDF (first time only)...")
def build_embedding_index():
    """
    1) Load PDF text
    2) Chunk it
    3) Create embeddings for each chunk
    Returns: (chunks_list, embeddings_matrix)
    """
    raw_text = load_pdf_text(PDF_PATH)
    chunks = chunk_text(raw_text)

    # If the PDF is big, this might call the API many times -> RateLimitError
    # We keep it to a single request here for simplicity.
    try:
        response = client.embeddings.create(
            model=EMBED_MODEL,
            input=chunks
        )
    except RateLimitError as e:
        # Let Streamlit show the error nicely
        raise RateLimitError(
            f"Rate limit hit while creating embeddings. "
            f"Your account/plan is limiting embedding calls. "
            f"Try again later or use a smaller PDF.\n\n{e}"
        )

    embeddings = np.array([d.embedding for d in response.data])
    return chunks, embeddings


def cosine_similarities(query_vec: np.ndarray, mat: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between a query vector and a matrix of vectors."""
    # (n_embeds, dim) · (dim,) -> (n_embeds,)
    dots = mat @ query_vec
    mat_norms = np.linalg.norm(mat, axis=1)
    query_norm = np.linalg.norm(query_vec) + 1e-8
    sims = dots / (mat_norms * query_norm + 1e-8)
    return sims


def retrieve_context(question: str, top_k: int = 4) -> str:
    """Embed the question, compare with PDF chunks, and return top_k chunks."""
    chunks, embeddings = build_embedding_index()

    # Embed the question
    q_emb = client.embeddings.create(
        model=EMBED_MODEL,
        input=[question]
    ).data[0].embedding
    q_emb = np.array(q_emb)

    sims = cosine_similarities(q_emb, embeddings)
    top_idx = sims.argsort()[-top_k:][::-1]  # highest similarity first

    selected_chunks = [chunks[i] for i in top_idx]
    return "\n\n---\n\n".join(selected_chunks)


# -------------------- ANSWER GENERATION --------------------
def get_rag_answer(question: str) -> str:
    """
    1) Retrieve relevant PDF chunks
    2) Send them + question to GPT
    3) Get final answer
    """
    try:
        context = retrieve_context(question)
    except RateLimitError as e:
        # If embeddings fail due to rate limit, fall back to pure AI answer
        context = (
            "NOTE: I could not access the company PDF because of an "
            "OpenAI rate limit / quota issue. I will answer using only "
            "my general knowledge."
        )

    system_prompt = f"""
You are **Zeeshan ka Chatbot**, a helpful, professional assistant.

You are retrieval-augmented:
- FIRST, rely on the PDF context from Zeeshan's company manual.
- SECOND, if the context is missing or not relevant, use general world knowledge.
- ALWAYS be honest about what you know and what you don't.
- If the question is about company policies and the PDF says something, follow the PDF strictly.

PDF CONTEXT (may be partial):

{context}

When you answer:
- Be clear and structured.
- Mention when you used the PDF (e.g., "According to the company manual...").
- If the PDF didn't help, say you are answering from general knowledge.
"""

    completion = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ],
    )

    return completion.choices[0].message.content


# -------------------- NICE UI --------------------
st.markdown(
    """
    <style>
    .title {
        font-size: 40px;
        font-weight: 800;
        background: linear-gradient(90deg, #00e0ff, #ff00ff);
        -webkit-background-clip: text;
        color: transparent;
    }
    .subtitle {
        font-size: 16px;
        color: #bbbbbb;
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="title">🤖 Zeeshan ka Chatbot (RAG)</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Powered by your company manual + GPT-4o-mini. '
    'Ask about policies, procedures, or anything else.</div>',
    unsafe_allow_html=True,
)

with st.expander("ℹ️ What this bot actually does (RAG):", expanded=False):
    st.write(
        """
        - Reads **Zeeshan_Chatbot_Company_Manual.pdf**
        - Breaks it into small chunks and creates **embeddings** (vector representation)
        - For each question, finds the **most similar PDF chunks**
        - Sends those chunks + your question to GPT-4o-mini
        - Answers using both **PDF knowledge** and **general AI knowledge**
        """
    )

# Session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show previous chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input (clears automatically)
user_prompt = st.chat_input("Ask Zeeshan ka Chatbot anything...")

if user_prompt:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Generate bot answer
    with st.chat_message("assistant"):
        placeholder = st.empty()
        try:
            answer = get_rag_answer(user_prompt)
        except Exception as e:
            answer = (
                "❌ An error occurred while generating the answer:\n\n"
                f"`{type(e).__name__}: {e}`\n\n"
                "If it's a rate limit or quota error, you need more OpenAI credits "
                "or fewer embedding calls."
            )
        placeholder.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
