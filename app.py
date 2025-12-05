import streamlit as st
from openai import OpenAI

# Load API key safely
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

st.set_page_config(page_title="AI Chatbot", layout="wide")

# ---------------------------
#   INITIAL SETUP
# ---------------------------

st.markdown(
    "<h1 style='text-align:center;'>🤖 AI Voice + Text Chatbot</h1>",
    unsafe_allow_html=True
)

if "messages" not in st.session_state:
    st.session_state["messages"] = []

# ---------------------------
#   DISPLAY CHAT HISTORY
# ---------------------------

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ---------------------------
#   VOICE INPUT (SAFE)
# ---------------------------

st.subheader("🎤 Speak your question")

audio_data = st.audio_input("Click to record...")

voice_text = ""

if audio_data is not None:
    st.success("Audio recorded — processing...")

    audio_bytes = audio_data.getvalue()

    #  -------- PROTECTION AGAINST TOO-LONG AUDIO -------
    if len(audio_bytes) > 1_500_000:  # about 1.5 MB ≈ 10–12 seconds
        st.warning("⚠️ Audio too long! Please record 10 seconds maximum.")
    else:
        #  -------- SAFE WHISPER CALL WITH ERROR HANDLING -------
        try:
            transcript = client.audio.transcriptions.create(
                model="gpt-4o-mini-transcribe",   # Faster + cheaper + higher limits
                file=audio_bytes
            )
            voice_text = transcript.text
            st.info(f"🗣️ You said: **{voice_text}**")

        except Exception as e:
            st.error("⚠️ Voice transcription failed due to rate limit or credits. Try again later.")
            st.stop()   # stops code safely

# ---------------------------
#   TEXT INPUT WITH AUTO-CLEAR
# ---------------------------

if "text_input" not in st.session_state:
    st.session_state["text_input"] = ""

def submit_text():
    st.session_state["submitted_text"] = st.session_state["text_input"]
    st.session_state["text_input"] = ""  # auto clear

st.text_input(
    "Type a message:",
    key="text_input",
    on_change=submit_text
)

text_message = st.session_state.get("submitted_text", "")

# ---------------------------
#   DETERMINE USER MESSAGE
# ---------------------------

final_user_message = ""

if voice_text:
    final_user_message = voice_text
elif text_message:
    final_user_message = text_message

# ---------------------------
#   GENERATE AI RESPONSE
# ---------------------------

if final_user_message:

    # Add user's message to history
    st.session_state["messages"].append(
        {"role": "user", "content": final_user_message}
    )

    # Display user message
    with st.chat_message("user"):
        st.write(final_user_message)

    # ----- SAFE CHATGPT CALL -----
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=st.session_state["messages"]
        )

        bot_reply = response.choices[0].message.content

    except Exception as e:
        bot_reply = "⚠️ Unable to generate a response due to API rate limit or server overload."

    # Save assistant reply
    st.session_state["messages"].append(
        {"role": "assistant", "content": bot_reply}
    )

    # Display assistant reply
    with st.chat_message("assistant"):
        st.write(bot_reply)
