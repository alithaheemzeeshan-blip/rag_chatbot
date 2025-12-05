import streamlit as st
from openai import OpenAI

# Load API key safely
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

st.set_page_config(page_title="Zeeshan Ka Chatbot", layout="wide")

# ---------------------------
#   CUSTOM UI STYLING
# ---------------------------
st.markdown("""
    <style>
        body {
            background-color: #0E1117;
        }
        .stApp {
            background-color: #0E1117;
        }
        .main-title {
            font-size: 42px;
            font-weight: 800;
            text-align: center;
            color: #00C4FF;
            margin-top: -20px;
            text-shadow: 0px 0px 15px rgba(0, 196, 255, 0.6);
        }
        .sub-title {
            font-size: 20px;
            text-align: center;
            color: #cccccc;
            margin-bottom: 30px;
        }
        .chat-bubble-user {
            background-color: #1F2937;
            padding: 12px;
            border-radius: 12px;
            color: white;
            margin: 6px 0;
        }
        .chat-bubble-bot {
            background-color: #111827;
            padding: 12px;
            border-radius: 12px;
            color: white;
            border-left: 3px solid #00C4FF;
            margin: 6px 0;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-title'>🤖 Zeeshan Ka Chatbot</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Aap ka personal AI assistant — hamesha madad ko tayyar!</p>", unsafe_allow_html=True)

# ---------------------------
#   CHAT MEMORY
# ---------------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# ---------------------------
#   DISPLAY CHAT HISTORY (styled)
# ---------------------------
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        bubble = "chat-bubble-user" if msg["role"] == "user" else "chat-bubble-bot"
        st.markdown(f"<div class='{bubble}'>{msg['content']}</div>", unsafe_allow_html=True)

# ---------------------------
#   🎤 VOICE INPUT (SAFE)
# ---------------------------
st.subheader("🎤 Aap bol kar bhi sawaal pooch saktay hain:")

audio_data = st.audio_input("Record your voice:")

voice_text = ""

if audio_data is not None:
    st.success("Audio received — processing...")

    audio_bytes = audio_data.getvalue()

    # Too long audio protection
    if len(audio_bytes) > 1_500_000:
        st.warning("⚠️ Audio bohat lamba hai! Please 10 seconds se kam record karein.")
    else:
        try:
            transcript = client.audio.transcriptions.create(
                model="gpt-4o-mini-transcribe",
                file=audio_bytes
            )
            voice_text = transcript.text
            st.info(f"🗣️ Aap ne kaha: **{voice_text}**")

        except Exception:
            st.error("⚠️ Voice transcription failed due to rate limit or credit issue.")
            voice_text = ""

# ---------------------------
#   TEXT INPUT (AUTO CLEAR)
# ---------------------------
if "text_input" not in st.session_state:
    st.session_state["text_input"] = ""

def submit_text():
    st.session_state["submitted_text"] = st.session_state["text_input"]
    st.session_state["text_input"] = "" 

st.text_input(
    "✍️ Type your message here:",
    key="text_input",
    on_change=submit_text
)

text_message = st.session_state.get("submitted_text", "")

# Decide final user message
final_user_message = voice_text if voice_text else text_message

# ---------------------------
#   RESPONSE GENERATION
# ---------------------------
if final_user_message:

    # Save user message
    st.session_state["messages"].append(
        {"role": "user", "content": final_user_message}
    )

    # Show user bubble
    with st.chat_message("user"):
        st.markdown(f"<div class='chat-bubble-user'>{final_user_message}</div>", unsafe_allow_html=True)

    # Call GPT
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=st.session_state["messages"]
        )
        bot_reply = response.choices[0].message.content

    except Exception:
        bot_reply = "⚠️ Sorry, API busy hai. Thori der baad try karein."

    # Save bot reply
    st.session_state["messages"].append(
        {"role": "assistant", "content": bot_reply}
    )

    # Show bot bubble
    with st.chat_message("assistant"):
        st.markdown(f"<div class='chat-bubble-bot'>{bot_reply}</div>", unsafe_allow_html=True)
