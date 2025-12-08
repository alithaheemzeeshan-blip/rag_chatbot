import streamlit as st
import pdfplumber
from openai import OpenAI


# -----------------------------
# OPENAI CLIENT
# -----------------------------
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


# -----------------------------
# LOAD & CHUNK PDF
# -----------------------------
def load_pdf_chunks(chunk_size=600):
    pdf_path = "Zeeshan_Chatbot_Company_Manual.pdf"
    text = ""

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for p in pdf.pages:
                extracted = p.extract_text()
                if extracted:
                    text += extracted + "\n"
    except:
        return ["(PDF missing — no internal data available.)"]

    # Simple chunking
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks


CHUNKS = load_pdf_chunks()


# -----------------------------
# SIMPLE KEYWORD RETRIEVAL (NAIVE RAG)
# -----------------------------
def retrieve_relevant_chunk(query):
    query_words = query.lower().split()
    best_chunk = ""
    best_score = 0

    for chunk in CHUNKS:
        score = sum(chunk.lower().count(word) for word in query_words)
        if score > best_score:
            best_score = score
            best_chunk = chunk

    return best_chunk if best_chunk else "(No matching info found in PDF.)"


# -----------------------------
# GENERATE ANSWER USING RAG
# -----------------------------
def get_answer(query):
    context = retrieve_relevant_chunk(query)

    system_message = f"""
You are **Zeeshan ka Chatbot**, a simple RAG-based assistant.

Use the following retrieved context from the company manual to answer:

CONTEXT:
{context}

If the context does not answer, say:
"The PDF does not contain this information. Here is general guidance:"
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": query},
        ],
    )

    return response.choices[0].message["content"]


# -----------------------------
# STREAMLIT UI
# -----------------------------
st.set_page_config(page_title="Zeeshan ka Chatbot", page_icon="🤖")

st.title("🤖 Zeeshan ka Chatbot (Simple RAG)")
st.write("Ask anything from your company manual or general knowledge.")


if "history" not in st.session_state:
    st.session_state.history = []

user_input = st.text_input("Ask something:", key="input_box")

if st.button("Send"):
    if user_input.strip() != "":
        answer = get_answer(user_input)
        st.session_state.history.append(("You", user_input))
        st.session_state.history.append(("Bot", answer))
        st.session_state.input_box = ""  # CLEAR INPUT BOX


# Display chat
for sender, msg in st.session_state.history:
    if sender == "You":
        st.markdown(f"**🧑 You:** {msg}")
    else:
        st.markdown(f"**🤖 Zeeshan ka Chatbot:** {msg}")
