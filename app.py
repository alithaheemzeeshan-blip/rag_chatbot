import streamlit as st
from openai import OpenAI
import pdfplumber

# ---------------------------
# 🔑 OPENAI CLIENT
# ---------------------------
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# ---------------------------
# 📄 LOAD PDF KNOWLEDGE
# ---------------------------
PDF_PATH = "Zeeshan_Chatbot_Company_Manual.pdf"

def load_pdf_text():
    text = ""
    try:
        with pdfplumber.open(PDF_PATH) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"
    except:
        text = "Could not load PDF content."
    
    return text

PDF_TEXT = load_pdf_text()

# ---------------------------
# 🤖 GENERATE ANSWER (FIXED)
# ---------------------------
def generate_answer(question):
    full_context = f"""
    Company Manual Knowledge:
    {PDF_TEXT}

    User Question:
    {question}

    Respond using the company PDF first. If answer not found, use general AI knowledge.
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are Zeeshan ka Chatbot — a corporate assistant with a tech-robot personality."},
            {"role": "user", "content": full_context},
        ]
    )

    # FIX: correct API format
    return response.choices[0].message.content


# ---------------------------
# 🎨 STREAMLIT UI
# ---------------------------
st.set_page_config(page_title="Zeeshan ka Chatbot", page_icon="🤖", layout="centered")

st.markdown("<h1 style='text-align:center;'>🤖 Zeeshan ka Chatbot</h1>", unsafe_allow_html=True)
st.write("<p style='text-align:center;'>Your corporate AI assistant powered by RAG + PDF knowledge.</p>", unsafe_allow_html=True)

st.divider()

# ---------------------------
# 💬 CHAT LOG
# ---------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------------------------
# 🧑‍💻 USER INPUT
# ---------------------------
user_input = st.text_input("Ask something:", key="input_box")

if st.button("Send"):
    if user_input.strip() != "":
        answer = generate_answer(user_input)

        # Save chat
        st.session_state.chat_history.append(
            {"q": user_input, "a": answer}
        )

        # Clear input box
        st.session_state.input_box = ""

# ---------------------------
# 📥 DISPLAY CHAT MESSAGES
# ---------------------------
for msg in st.session_state.chat_history:
    st.markdown(f"""
        <div style='padding:10px; background:#1e1e1e; border-radius:8px; margin-bottom:8px;'>
            <b>🧑 You:</b> {msg['q']}
        </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
        <div style='padding:10px; background:#0b3d91; border-radius:8px; margin-bottom:15px; color:white;'>
            <b>🤖 Zeeshan ka Chatbot:</b> {msg['a']}
        </div>
    """, unsafe_allow_html=True)
