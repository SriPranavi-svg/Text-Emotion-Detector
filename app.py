
import streamlit as st
from transformers import pipeline

# Page configuration
st.set_page_config(
    page_title="Text Emotion Detector",
    page_icon="😊",
    layout="centered"
)

# Load the AI model
@st.cache_resource
def load_model():
    return pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base"
    )

# Title and description
st.title("😊 Text Emotion Detector")
st.write("Enter a sentence and discover the emotion expressed in it.")

# Load model
with st.spinner("Loading AI model..."):
    classifier = load_model()

# Text input
user_text = st.text_area(
    "Enter your text:",
    placeholder="Example: I am very happy today!"
)

# Emotion emojis
emojis = {
    "anger": "😠",
    "disgust": "🤢",
    "fear": "😨",
    "joy": "😊",
    "neutral": "😐",
    "sadness": "😢",
    "surprise": "😲"
}

# Detect emotion
if st.button("Detect Emotion"):
    if user_text.strip():
        with st.spinner("Analyzing your text..."):
            result = classifier(user_text)[0]

        emotion = result["label"]
        confidence = result["score"] * 100

        st.subheader("Prediction Result")

        st.success(
            f"{emojis.get(emotion, '🙂')} Emotion: "
            f"{emotion.capitalize()}"
        )

        st.write(f"Confidence score: {confidence:.2f}%")

        st.progress(result["score"])

        st.caption(
            "This is an AI prediction and may not always "
            "reflect the intended emotion."
        )
    else:
        st.warning("Please enter some text first.")
