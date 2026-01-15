import os
import re
import gradio as gr
import google.generativeai as genai
import matplotlib.pyplot as plt
from datetime import datetime
from collections import defaultdict

API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
if not API_KEY:
    raise RuntimeError(
        "Missing GEMINI_API_KEY. Set it as an environment variable before running.\n"
        "Example (Linux/macOS): export GEMINI_API_KEY='YOUR_KEY'\n"
        "Example (Windows PowerShell): setx GEMINI_API_KEY 'YOUR_KEY'"
    )

genai.configure(api_key=API_KEY)

MODEL_NAME = "gemini-1.5-pro-latest"

emotion_values_by_day = defaultdict(list)

EMOTION_MAP = {"Happy 😊": 3, "Neutral 😐": 2, "Anxious 😟": 1, "Sad 😔": 0}
REVERSE_EMOTION_MAP = {v: k for k, v in EMOTION_MAP.items()}

def detect_emotion(text: str) -> str:
    """
    Simple keyword-based emotion detection.
    You can upgrade this later (e.g., ask the model to classify).
    """
    t = text.lower()

   
    happy = ["great", "good", "happy", "joy", "excited", "relieved", "grateful", "proud"]
    sad = ["sad", "down", "unhappy", "depressed", "hopeless", "lonely", "empty", "tear"]
    anxious = ["nervous", "worried", "anxious", "stressed", "panic", "overwhelmed", "tense"]

    if any(w in t for w in happy):
        return "Happy 😊"
    if any(w in t for w in sad):
        return "Sad 😔"
    if any(w in t for w in anxious):
        return "Anxious 😟"
    return "Neutral 😐"

def update_emotion_history(ai_text: str):
    today = datetime.today().strftime("%Y-%m-%d")
    emotion_label = detect_emotion(ai_text)
    emotion_values_by_day[today].append(EMOTION_MAP[emotion_label])

def generate_emotion_chart():
    """
    Returns a filepath to a saved PNG chart, or None if no data.
    Gradio Image can accept a filepath string.
    """
    if not emotion_values_by_day:
        return None

   
    dates = sorted(emotion_values_by_day.keys())

    
    daily_avg = []
    for d in dates:
        vals = emotion_values_by_day[d]
        avg = sum(vals) / len(vals)
        daily_avg.append(avg)

    plt.figure(figsize=(9, 4))
    plt.plot(dates, daily_avg, marker="o", linestyle="-", linewidth=2)
    plt.xticks(rotation=45, ha="right")

    plt.yticks([0, 1, 2, 3], ["Sad 😔", "Anxious 😟", "Neutral 😐", "Happy 😊"])
    plt.ylim(-0.2, 3.2)

    plt.title("Daily Emotion Trend 📈 (average per day)")
    plt.xlabel("Date")
    plt.ylabel("Emotion Level")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = "emotion_chart.png"
    plt.savefig(out_path, dpi=160)
    plt.close()
    return out_path


CRISIS_PATTERN = re.compile(
    r"\b(suicide|kill myself|end my life|self harm|self-harm|hurt myself|want to die)\b",
    re.IGNORECASE
)

def crisis_message():
    return (
        "I'm really sorry you're feeling this way. If you're in immediate danger or feel like you might harm yourself, "
        "please call your local emergency number right now.\n\n"
        "UK & ROI: Samaritans 116 123 (free, 24/7)\n"
        "US & Canada: Call or text 988 (Suicide & Crisis Lifeline)\n"
        "Elsewhere: Tell me your country and I can suggest local options.\n\n"
        "If you can, try not to be alone right now—reach out to someone you trust."
    )



SYSTEM_STYLE = (
    "You are a supportive mental health companion. Be warm, non-judgmental, and practical. "
    "Use short paragraphs. Ask gentle questions. Offer coping strategies (breathing, grounding, journaling, routine). "
    "Do not claim to be a therapist. If user mentions self-harm or suicide, encourage immediate professional help."
)

def chat_with_ai(user_input, history):
    """
    Gradio Chatbot expects `history` as list of (user, assistant) tuples.
    We return updated history.
    """
    user_input = (user_input or "").strip()
    history = history or []

    if not user_input:
        history.append(("", "Please enter a message so I can help."))
        return history, ""

    
    if CRISIS_PATTERN.search(user_input):
        history.append((user_input, crisis_message()))
        return history, ""

    try:
        model = genai.GenerativeModel(MODEL_NAME)

        prompt = (
            f"{SYSTEM_STYLE}\n\n"
            f"User says: {user_input}\n\n"
            "Respond with empathy, then 1–3 practical steps. End with one gentle question."
        )

        response = model.generate_content(prompt)

        ai_text = (getattr(response, "text", None) or "").strip()
        if not ai_text:
            ai_text = "I’m here with you. I couldn’t generate a full response—could you try rephrasing that?"

       
        update_emotion_history(ai_text)

        
        history.append((user_input, ai_text))
        return history, ""

    except Exception as e:
        history.append((user_input, f"Error while generating response: {e}"))
        return history, ""

def on_show_chart():
    path = generate_emotion_chart()
    if not path:
        return None, "No emotion data yet. Chat a bit first, then try again."
    return path, ""

def clear_all():
    emotion_values_by_day.clear()
    return [], None, ""


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🌿 Mental Health Companion")
    gr.Markdown(
        "Talk to an AI that provides **supportive guidance** for your mental well-being. 💙  \n"
        "_If you're in immediate danger or thinking about self-harm, please contact local emergency services._"
    )

    chatbot = gr.Chatbot(height=420, bubble_full_width=False)

    with gr.Row():
        user_input = gr.Textbox(
            placeholder="How are you feeling today?",
            label="",
            lines=2
        )

    with gr.Row():
        send_btn = gr.Button("Send ✨", variant="primary")
        chart_btn = gr.Button("Show Emotion Chart 📊")
        clear_btn = gr.Button("Clear Chat & Data 🧹")

    with gr.Row():
        emotion_chart = gr.Image(label="Emotion Trend", height=320)
    status = gr.Markdown("")

    send_btn.click(chat_with_ai, inputs=[user_input, chatbot], outputs=[chatbot, user_input])
    user_input.submit(chat_with_ai, inputs=[user_input, chatbot], outputs=[chatbot, user_input])

    chart_btn.click(on_show_chart, inputs=None, outputs=[emotion_chart, status])
    clear_btn.click(clear_all, inputs=None, outputs=[chatbot, emotion_chart, status])

demo.launch(server_name="0.0.0.0", server_port=7860)
