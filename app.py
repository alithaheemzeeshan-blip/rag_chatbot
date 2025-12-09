import streamlit as st
import pdfplumber
from groq import Groq
from tavily import TavilyClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="Zeeshan RAG Chatbot", layout="centered")

# -------------------- API KEYS --------------------
GROQ_KEY = st.secrets["GROQ_API_KEY"]
TAVILY_KEY = st.secrets["TAVILY_API_KEY"]

client = Groq(api_key=GROQ_KEY)
tavily = TavilyClient(api_key=TAVILY_KEY)

PDF_PATH = "data/Zeeshan_Chatbot_Company_Manual.pdf"


# -------------------- LOAD PDF --------------------
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


# -------------------- SPLIT INTO CHUNKS --------------------
@st.cache_data
def create_chunks(text, chunk_size=600):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

chunks = create_chunks(pdf_text)


# -------------------- TF-IDF VECTOR SEARCH --------------------
@st.cache_data
def build_tfidf(chunks):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(chunks)
    return vectorizer, vectors

vectorizer, vectors = build_tfidf(chunks)


def retrieve_pdf_context(query):
    """Return top chunk from PDF using cosine similarity"""
    q_vec = vectorizer.transform([query])
    scores = cosine_similarity(q_vec, vectors).flatten()
    top_index = scores.argmax()
    return chunks[top_index]


# -------------------- INTERNET SEARCH --------------------
def search_web(query):
    response = tavily.search(query=query, max_results=3)
    results = "\n".join([r["content"] for r in response["results"]])
    return results if results else "No live results found."


# -------------------- LLAMA ANSWER --------------------
def generate_answer(user_query):
    pdf_context = retrieve_pdf_context(user_query)
    web_context = search_web(user_query)

    prompt = f"""
You are Zeeshan's AI Chatbot.

Use BOTH sources to answer accurately and with latest information.

PDF CONTEXT:
{pdf_context}

LIVE INTERNET SEARCH:
{web_context}

Now answer the user's question clearly.
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_query}
        ]
    )

    return response.choices[0].message["content"]


# -------------------- UI --------------------
st.title("🤖 Zeeshan RAG Chatbot (PDF + Live Internet Search)")

if "chat" not in st.session_state:
    st.session_state.chat = []

with st.form("chat_form", clear_on_submit=True):
    q = st.text_input("Ask me anything:")
    send = st.form_submit_button("Send")

if send and q.strip():
    answer = generate_answer(q)
    st.session_state.chat.append(("You", q))
    st.session_state.chat.append(("Bot", answer))

# Show chat
st.write("---")
for role, msg in st.session_state.chat:
    if role == "You":
        st.markdown(f"🧑 **You:** {msg}")
    else:
        st.markdown(f"🤖 **Bot:** {msg}")
