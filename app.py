import streamlit as st
import pdfplumber
from openai import OpenAI
import numpy as np

st.set_page_config(page_title="Zeeshan ka Chatbot", layout="centered")

# Load API Key
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

PDF_PATH = "data/Zeeshan_Chatbot_Company_Manual.pdf"

# -------------------------------------------
# 1. LOAD PDF
# -------------------------------------------
@st.cache_data
def load_pdf():
    text = ""
    with pdfplumber.open(PDF_PATH) as pdf:
        for page in pdf.pages:
            tx = page.extract_text()
            if tx:
                text += tx + "\n"
    return text

raw_text = load_pdf()


# -------------------------------------------
# 2. SPLIT PDF TEXT INTO CHUNKS
# -------------------------------------------
def split_into_chunks(text, chunk_size=500):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

chunks = split_into_chunks(raw_text)


# -------------------------------------------
# 3. CREATE EMBEDDINGS FOR EACH CHUNK
# -------------------------------------------
@st.cache_data
def embed_chunks(chunk_list):
    embed_list = []
    for chunk in chunk_list:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=chunk
        )
        embed_list.append(response.data[0].embedding)
    return np.array(embed_list)

chunk_embeddings = embed_chunks(chunks)


# -------------------------------------------
# 4. FIND MOST RELEVANT CHUNKS
# -------------------------------------------
def retrieve_context(query, top_k=3):
    query_emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    ).data[0].embedding

    query_emb = np.array(query_emb)

    # cosine similarity
    similarities = np.dot(chunk_embeddings, query_emb)

    top_indices = similarities.argsort()[::-1][:top_k]

    retrieved = "\n\n".join([chunks[i] for i in top_indices])
    return retrieved


# -------------------------------------------
# 5. GET AI ANSWER USING RETRIEVED CONTEXT
# -------------------------------------------
def get_answer(user_question):
    context = retrieve_context(user_question)

    system_prompt = f"""
You are Zeeshan ka Chatbot.
You answer ONLY using the PDF knowledge below:

PDF CONTEXT:
{context}

If the answer is not present in the PDF, say:
"Sorry, this information is not available in my company manual."
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_question}
        ]
    )

    return response.choices[0].message.content


# -------------------------------------------
# 6. USER INTERFACE
# -------------------------------------------
st.title("🤖 Zeeshan ka RAG Chatbot")

if "chat" not in st.session_state:
    st.session_state.chat = []

with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("Ask something:")
    submitted = st.form_submit_button("Send")

if submitted and user_input.strip():
    answer = get_answer(user_input)
    st.session_state.chat.append(("You", user_input))
    st.session_state.chat.append(("Bot", answer))

st.write("---")
for sender, msg in st.session_state.chat:
    if sender == "You":
        st.markdown(f"**🧑 You:** {msg}")
    else:
        st.markdown(f"**🤖 Zeeshan ka Chatbot:** {msg}")
