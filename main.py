import gradio as gr
import google.generativeai as genai
import matplotlib.pyplot as plt
from datetime import datetime

# Replace with your actual Google Gemini API key
genai.configure(api_key="AIzaSyBh4dr45WvZSPcDyqonHatUv850P_K4iZA")

# Emotion tracking dictionary
emotion_history = {}

# Function to detect emotion from AI response
def detect_emotion(response):
    response = response.lower()
    if any(word in response for word in ["great", "good", "happy", "joy", "excited"]):
        return "Happy 😊"
    elif any(word in response for word in ["sad", "down", "unhappy", "depressed", "hopeless"]):
        return "Sad 😔"
    elif any(word in response for word in ["nervous", "worried", "anxious", "stressed"]):
        return "Anxious 😟"
    else:
        return "Neutral 😐"

# Chat function
def chat_with_ai(user_input, chat_history):
    if not user_input.strip():
        return chat_history + [("Please enter a valid message.", "")]

    try:
        model = genai.GenerativeModel("gemini-1.5-pro-latest")
        response = model.generate_content(user_input)

        if response.text:
            chat_history.append((user_input, ""))  # User message (right-aligned)
            chat_history.append(("", response.text))  # AI message (left-aligned)
            
            # Store detected emotion
            today = datetime.today().strftime('%Y-%m-%d')
            emotion_history[today] = detect_emotion(response.text)

            return chat_history
        else:
            return chat_history + [("", "I couldn't generate a response.")]
    
    except Exception as e:
        return chat_history + [("", f"Error: {str(e)}")]

# Function to generate emotion trend line chart
def generate_emotion_chart():
    if not emotion_history:
        return "No emotion data available yet."

    dates = list(emotion_history.keys())
    emotions = list(emotion_history.values())

    # Map emotions to numbers for plotting
    emotion_map = {"Happy 😊": 3, "Neutral 😐": 2, "Anxious 😟": 1, "Sad 😔": 0}
    emotion_values = [emotion_map[emotion] for emotion in emotions]

    # Plot the line chart
    plt.figure(figsize=(8, 4))
    plt.plot(dates, emotion_values, marker="o", linestyle="-", color="blue", linewidth=2, markersize=8)
    plt.xticks(rotation=45)
    plt.yticks([0, 1, 2, 3], ["Sad 😔", "Anxious 😟", "Neutral 😐", "Happy 😊"])
    plt.title("Daily Emotion Trend 📈")
    plt.xlabel("Date")
    plt.ylabel("Emotion Level")
    plt.grid(True)
    
    # Save and return the chart
    plt.savefig("emotion_chart.png")
    plt.close()
    return "emotion_chart.png"

# Gradio UI
with gr.Blocks(theme="soft") as demo:
    gr.Markdown("# 🌿 Mental Health Companion")
    gr.Markdown("Talk to an AI that provides **support and guidance** for your mental well-being. 💙")

    with gr.Row():
        chat_box = gr.Chatbot(height=400, bubble_full_width=False)  # Chat layout

    with gr.Row():
        user_input = gr.Textbox(placeholder="How are you feeling today?", label="", interactive=True)
        send_button = gr.Button("Send ✨")
    
    with gr.Row():
        chart_button = gr.Button("Show Emotion Chart 📊")
    
    emotion_chart = gr.Image()

    send_button.click(chat_with_ai, inputs=[user_input, chat_box], outputs=chat_box)
    chart_button.click(generate_emotion_chart, outputs=emotion_chart)

# Launch the app
demo.launch(server_name="0.0.0.0", server_port=7860)
