import streamlit as st
import openai
from streamlit_chat import message

st.set_page_config(page_title="Voice ChatGPT", page_icon="🎤", layout="centered")

openai.api_key = st.secrets["OPENAI_API_KEY"]

# ------- CUSTOM UI CSS --------
st.markdown("""
<style>
.chat-box {
    background-color: #202123;
    padding: 20px;
    border-radius: 15px;
    width: 100%;
    max-width: 750px;
    margin: auto;
}
.stTextInput>div>div>input {
    background: #3A3B3C;
    color: white;
}
.voice-button {
    background-color:#4a90e2;
    padding:10px 20px;
    color:white;
    border-radius:10px;
    cursor:pointer;
    margin-top:10px;
    display:inline-block;
}
</style>
""", unsafe_allow_html=True)

# ------- JAVASCRIPT FOR FULL BROWSER VOICE INPUT --------
voice_js = """
<script>
let recognition;

function startRecognition() {
    const textarea = window.parent.document.querySelector('textarea');

    if (!('webkitSpeechRecognition' in window)) {
        alert("Your browser does not support Speech Recognition.");
    } else {
        recognition = new webkitSpeechRecognition();
        recognition.continuous = false;
        recognition.interimResults = false;
        recognition.lang = "en-US";

        recognition.start();

        recognition.onresult = function(event) {
            const text = event.results[0][0].transcript;
            textarea.value = text;
            textarea.dispatchEvent(new Event('input', { bubbles: true }));
        };
    }
}
</script>

<div class="voice-button" onclick="startRecognition()">🎤 Speak</div>
"""

# ------- GPT FUNCTION --------
def ask_gpt(prompt):
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# ------- SESSION STATE --------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ------- UI --------
st.title("🎤 Voice ChatGPT (Full Browser Voice Support)")

st.markdown('<div class="chat-box">', unsafe_allow_html=True)

# Show chat history
for msg in st.session_state.messages:
    message(msg["content"], is_user=(msg["role"] == "user"))

# ------- TEXT INPUT --------
user_text = st.text_input("Ask me something:")

# Voice Button Added Using HTML + JS
st.components.v1.html(voice_js, height=80)

if user_text:
    st.session_state.messages.append({"role": "user", "content": user_text})
    bot_reply = ask_gpt(user_text)
    st.session_state.messages.append({"role": "assistant", "content": bot_reply})
    st.rerun()

st.markdown('</div>', unsafe_allow_html=True)
