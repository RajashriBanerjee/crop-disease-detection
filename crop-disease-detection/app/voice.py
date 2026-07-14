"""
Accessibility layer: lets someone who can't read or write use the app —
results are spoken aloud in a local language of their choice, and they can
also speak a crop/disease name instead of typing to look it up.

Three capabilities, each isolated so a missing package or a dropped network
call degrades gracefully instead of crashing the app:
    - translate_text()   : translate English text into the chosen language
    - text_to_speech()   : turn text into playable audio (mp3 bytes)
    - transcribe_audio() : turn a recorded voice clip into text

Both translation and speech synthesis call external services (Google
Translate / Google Text-to-Speech via gTTS) and need internet access at
runtime — same as any hosted Streamlit app talking to an external API.
"""

import difflib
import io

# Language code -> label shown in the UI. Codes are gTTS/Google Translate
# language codes. This list leans toward languages widely spoken across
# Indian farming communities, since that's the deployment context this was
# built for — swap/extend LANGUAGES for other regions.
LANGUAGES = {
    "en": "English",
    "hi": "Hindi (हिंदी)",
    "bn": "Bengali (বাংলা)",
    "ta": "Tamil (தமிழ்)",
    "te": "Telugu (తెలుగు)",
    "mr": "Marathi (मराठी)",
    "kn": "Kannada (ಕನ್ನಡ)",
    "gu": "Gujarati (ગુજરાતી)",
    "pa": "Punjabi (ਪੰਜਾਬੀ)",
}

# gTTS uses plain language codes; Google's speech recognizer wants a
# locale (e.g. "hi-IN"). Map our language codes to a reasonable locale.
SPEECH_LOCALES = {
    "en": "en-IN",
    "hi": "hi-IN",
    "bn": "bn-IN",
    "ta": "ta-IN",
    "te": "te-IN",
    "mr": "mr-IN",
    "kn": "kn-IN",
    "gu": "gu-IN",
    "pa": "pa-IN",
}


def translate_text(text: str, target_lang: str) -> str:
    """Translate English text into target_lang. Falls back to the original
    text if translation is unavailable, so the app still works without it."""
    if target_lang == "en" or not text:
        return text
    try:
        from deep_translator import GoogleTranslator

        return GoogleTranslator(source="en", target=target_lang).translate(text)
    except Exception:
        return text  # network hiccup or package missing — don't break the app


def text_to_speech(text: str, lang: str):
    """Synthesize speech for text in the given language.
    Returns mp3 bytes, or None if synthesis fails/unavailable."""
    if not text:
        return None
    try:
        from gtts import gTTS

        buf = io.BytesIO()
        gTTS(text=text, lang=lang).write_to_fp(buf)
        buf.seek(0)
        return buf.read()
    except Exception:
        return None


def transcribe_audio(audio_bytes: bytes, lang: str = "en"):
    """Transcribe a recorded voice clip (WAV bytes, as produced by
    st.audio_input) to text using the given language's locale.
    Returns the transcribed string, or None if it couldn't be understood."""
    if not audio_bytes:
        return None
    try:
        import speech_recognition as sr

        recognizer = sr.Recognizer()
        with sr.AudioFile(io.BytesIO(audio_bytes)) as source:
            audio_data = recognizer.record(source)
        locale = SPEECH_LOCALES.get(lang, "en-IN")
        return recognizer.recognize_google(audio_data, language=locale)
    except Exception:
        return None


def find_best_class_match(query: str, class_names: list):
    """Fuzzy-match a spoken/typed query (e.g. "tomato early blight") against
    the model's class names (e.g. "Tomato___Early_blight") so voice search
    doesn't need an exact string match. Returns the best class name or None."""
    if not query:
        return None

    # Build a lookup of human-readable name -> raw class name.
    readable_to_raw = {}
    for raw in class_names:
        crop, _, condition = raw.partition("___")
        readable = f"{crop.replace('_', ' ')} {condition.replace('_', ' ')}".strip().lower()
        readable_to_raw[readable] = raw

    matches = difflib.get_close_matches(query.lower(), readable_to_raw.keys(), n=1, cutoff=0.55)
    if matches:
        return readable_to_raw[matches[0]]
    return None
