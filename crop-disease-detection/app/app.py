"""
Crop Disease Detection demo app.

Wraps the same predict_disease() logic from the training notebook in a
Streamlit UI: upload a leaf photo, get the predicted disease, confidence,
and a treatment recommendation — spoken aloud in a local language, and
searchable by voice, for users who can't read or write.

Run locally:
    streamlit run app.py

Expects two files (produced by Step 9 of the notebook) inside ./models/:
    models/crop_disease_model.keras
    models/class_names.json
"""

import json
import os

import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.preprocessing import image as keras_image

from treatments import get_treatment
from voice import (
    LANGUAGES,
    find_best_class_match,
    text_to_speech,
    transcribe_audio,
    translate_text,
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "crop_disease_model.keras")
CLASS_NAMES_PATH = os.path.join(os.path.dirname(__file__), "models", "class_names.json")
IMG_SIZE = (224, 224)

st.set_page_config(page_title="Crop Disease Detector", page_icon="🌱", layout="centered")


@st.cache_resource(show_spinner="Loading model…")
def load_model_and_classes():
    if not os.path.exists(MODEL_PATH):
        return None, None
    import tensorflow as tf

    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASS_NAMES_PATH) as f:
        class_names = json.load(f)
    return model, class_names


def predict_disease(img: Image.Image, model, class_names, img_size=IMG_SIZE):
    """Same logic as the notebook's predict_disease(), adapted to take a
    PIL image directly (Streamlit's file_uploader gives us bytes, not a
    path) instead of a file path."""
    img_resized = img.convert("RGB").resize(img_size)
    img_array = keras_image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array, verbose=0)[0]
    top_idx = np.argmax(predictions)
    confidence = float(predictions[top_idx]) * 100

    # Top-3 for a bit more transparency in the demo than just the winner.
    top3_idx = np.argsort(predictions)[-3:][::-1]
    top3 = [(class_names[i], float(predictions[i]) * 100) for i in top3_idx]

    return class_names[top_idx], confidence, top3


def format_class_name(raw: str) -> str:
    crop, _, condition = raw.partition("___")
    crop = crop.replace("_", " ")
    condition = condition.replace("_", " ").strip()
    return f"{crop} — {condition}" if condition else crop


def render_result(predicted_class: str, confidence, lang: str, key_prefix: str, top3=None):
    """Shared result renderer used by both the photo-upload flow and the
    voice-search flow: shows the prediction, translates it into the chosen
    language, and offers a "listen" button that speaks it aloud — this is
    the core of making results usable for someone who can't read them."""
    info = get_treatment(predicted_class)
    is_healthy = predicted_class.endswith("healthy")
    readable_name = format_class_name(predicted_class)

    name_t = translate_text(readable_name, lang)
    desc_t = translate_text(info["description"], lang)
    treat_t = translate_text(info["treatment"], lang)

    if is_healthy:
        st.success(f"**{name_t}**")
    else:
        st.warning(f"**{name_t}**")
    if confidence is not None:
        st.metric("Confidence", f"{confidence:.1f}%")

    st.subheader("What this means" if lang == "en" else translate_text("What this means", lang))
    st.write(desc_t)

    st.subheader("Recommended action" if lang == "en" else translate_text("Recommended action", lang))
    st.write(treat_t)

    if top3:
        with st.expander("See top 3 predictions"):
            for name, conf in top3:
                st.write(f"{format_class_name(name)} — {conf:.1f}%")

    if st.button("🔊 Listen to result", key=f"{key_prefix}_listen"):
        spoken_text = f"{name_t}. {desc_t}. {treat_t}"
        with st.spinner("Generating audio…"):
            audio_bytes = text_to_speech(spoken_text, lang)
        if audio_bytes:
            st.audio(audio_bytes, format="audio/mp3")
        else:
            st.error(
                "Couldn't generate audio right now — check your internet "
                "connection (voice output needs it) and try again."
            )


# ---- UI ----

st.title("🌱 Crop Disease Detector")
st.caption(
    "Upload a photo of a crop leaf and get an instant disease prediction "
    "with a treatment recommendation, powered by a MobileNetV2 model "
    "fine-tuned on the PlantVillage dataset (38 classes, 97% validation accuracy)."
)

with st.sidebar:
    st.header("🌐 Language")
    lang_label = st.selectbox("Choose your language", options=list(LANGUAGES.values()), index=0)
    selected_lang = [code for code, label in LANGUAGES.items() if label == lang_label][0]
    st.caption(
        "Results can be read aloud in this language — helpful if you'd "
        "rather listen than read."
    )

model, class_names = load_model_and_classes()

if model is None:
    st.error(
        "Model files not found. Run the training notebook through Step 9 "
        "(`model.save('crop_disease_model.keras')`), then copy "
        "`crop_disease_model.keras` and `class_names.json` into "
        "`app/models/` before running this app."
    )
    st.stop()

tab_photo, tab_voice = st.tabs(["📷 Upload a photo", "🎤 Voice search"])

# ---- Tab 1: the original photo-upload flow ----
with tab_photo:
    uploaded_file = st.file_uploader("Upload a leaf photo", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        col1, col2 = st.columns([1, 1])

        with col1:
            st.image(img, caption="Uploaded image", use_container_width=True)

        with st.spinner("Analyzing leaf…"):
            predicted_class, confidence, top3 = predict_disease(img, model, class_names)

        with col2:
            render_result(predicted_class, confidence, selected_lang, key_prefix="photo", top3=top3)

        st.caption(
            "This is a demo model for a portfolio project, not agronomic advice. "
            "For real crop management decisions, consult a local agricultural "
            "extension office."
        )
    else:
        st.info("Upload a JPG or PNG of a single leaf to get a prediction.")

# ---- Tab 2: voice search — speak a crop/disease name instead of uploading ----
with tab_voice:
    st.write(
        "Don't have a photo, or would rather speak than type or read? "
        "Tap record and say a crop and disease name — for example "
        "*\"tomato early blight\"* or *\"apple healthy\"*."
    )
    st.caption(
        "Voice search needs an internet connection, since it uses an "
        "online speech-recognition service."
    )

    audio_value = st.audio_input("Record your search", key="voice_search_input")

    if audio_value is not None:
        with st.spinner("Listening…"):
            transcript = transcribe_audio(audio_value.read(), lang=selected_lang)

        if not transcript:
            st.error(
                "Couldn't understand that clearly — try again in a quiet "
                "space, or switch the language in the sidebar to match "
                "what you're speaking."
            )
        else:
            st.write(f"Heard: *\"{transcript}\"*")
            matched_class = find_best_class_match(transcript, class_names)

            if matched_class:
                render_result(matched_class, None, selected_lang, key_prefix="voice")
            else:
                st.warning(
                    "Couldn't match that to a known crop/disease. Try naming "
                    "the crop first, then the condition — e.g. \"potato late blight\"."
                )
