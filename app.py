import streamlit as st
import pdfplumber
from openai import OpenAI

# -------------------- SETTINGS --------------------
st.set_page_config(page_title="Zeeshan ka Chatbot", layout="centered")

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

PDF_PATH = "data/Zeeshan_Chatbot_Company_Manual.pdf"

# -------------------- LOAD PDF --------------------
@st.cache_data
def load_pdf_text():
    text = ""
    with pdfplumber.open(PDF_PATH) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

pdf_text = load_pdf_text()

# -------------------- SIMPLE RAG RETRIEVAL --------------------
def retrieve_relevant_context(query):
    """Find the most relevant chunk (very simple method)."""
    lines = pdf_text.split("\n")
    best = ""

    for line in lines:
        if query.lower() in line.lower():
            best = line
            break

    if best == "":
        best = "No direct match found in the PDF."

    return best


# -------------------- GET ANSWER --------------------
def get_answer(user_question):
    context = retrieve_relevant_context(user_question)

    system_prompt = f"""
You are **Zeeshan ka Chatbot**, a simple company helper bot.

Use the following PDF information to answer questions:

CONTEXT FROM PDF:
{context}

If the PDF does NOT give the answer, say:
"Ye baat PDF mein nahi hai, but here is a general answer:"
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_question}
        ]
    )

    return response.choices[0].message.content


# -------------------- UI --------------------
st.title("🤖 **Zeeshan ka Chatbot**")
st.write("Ask any question about your company manual!")

if "messages" not in st.session_state:
    st.session_state.messages = []

# USER INPUT
user_input = st.text_input("Ask something:", key="input_box")

if st.button("Send"):
    if user_input.strip() != "":
        answer = get_answer(user_input)

        # Save messages
        st.session_state.messages.append(("You", user_input))
        st.session_state.messages.append(("Bot", answer))

        # Clear input
        st.session_state.input_box = ""

# SHOW CHAT
st.write("---")
for sender, msg in st.session_state.messages:
    if sender == "You":
        st.markdown(f"**🧑 You:** {msg}")
    else:
        st.markdown(f"**🤖 Zeeshan ka Chatbot:** {msg}")
