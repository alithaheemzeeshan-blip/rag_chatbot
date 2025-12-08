import streamlit as st
import pdfplumber
from openai import OpenAI

# Load API key
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

st.set_page_config(page_title="Zeeshan ka Chatbot", layout="wide")

# ---------------------------
# Extract PDF text
# ---------------------------
def load_pdf_text():
    pdf_path = "data/Zeeshan_Chatbot_Company_Manual.pdf"
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
        return text
    except:
        return "PDF not found or failed to load."

pdf_text = load_pdf_text()

# ---------------------------
# RAG answer generator
# ---------------------------
def generate_answer(question):
    prompt = f"""
You are **Zeeshan ka Chatbot**, a professional AI assistant.
Use the following company manual to answer the question.

--- COMPANY DATA ---
{pdf_text}
---------------------

User question: {question}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are Zeeshan ka Chatbot. Answer politely and professionally."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content


# ---------------------------
# Streamlit UI
# ---------------------------
st.title("🤖 Zeeshan ka Chatbot")

st.write("### Ask something below:")

# Initialize message list
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Input box
user_input = st.text_input("Your question:", key="input_box", placeholder="Type here...")

# Send button
if st.button("Send"):
    if user_input.strip() != "":
        # Generate answer
        answer = generate_answer(user_input)

        # Save conversation
        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("Zeeshan ka Chatbot", answer))

        # Clear input box
        st.session_state.input_box = ""

# ---------------------------
# Display chat messages
# ---------------------------
st.write("### 💬 Chat History")

for sender, message in st.session_state.chat_history:
    if sender == "You":
        st.markdown(f"**🧑 You:** {message}")
    else:
        st.markdown(f"**🤖 Zeeshan ka Chatbot:** {message}")
