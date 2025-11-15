# app.py
import streamlit as st
import torch
import os
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from huggingface_hub import InferenceClient
from PIL import Image

# --- Page Configuration ---
st.set_page_config(
    page_title="GEN AI PROJECT 102397001",
    page_icon="📊",
    layout="centered",
    initial_sidebar_state="auto",
)

# --- Model and Client Loading (with Caching) ---
# @st.cache_resource is a powerful Streamlit feature. It loads the models only once,
# making the app much faster after the first run.
@st.cache_resource
def load_models():
    """Loads and caches the NLP models and the Inference Client."""
    # --- 1. Load Trained Sentiment Model and Translator (runs locally) ---
    st.write("Loading local NLP models... (This happens only on the first run)")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load sentiment pipeline
    sentiment_model_path = 'models/sentiment_classifier'
    sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_path)
    sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_path)
    sentiment_pipeline = pipeline(
        "sentiment-analysis", model=sentiment_model, tokenizer=sentiment_tokenizer,
        device=0 if device == "cuda" else -1
    )
    with open(f"{sentiment_model_path}/id_to_label.json", 'r', encoding='utf-8') as f:
        id_to_label = json.load(f)

    # Load translation pipeline
    translation_pipeline = pipeline(
        "translation", model="Helsinki-NLP/opus-mt-pa-en",
        device=0 if device == "cuda" else -1
    )
    
    # --- 2. Initialize the Hugging Face Inference Client ---
    hf_api_token = os.getenv("HF_TOKEN")
    if not hf_api_token:
        st.error("Hugging Face API token not found! Please set the HF_TOKEN environment variable.")
        return None, None, None, None

    inference_client = InferenceClient(token=hf_api_token)
    
    return sentiment_pipeline, id_to_label, translation_pipeline, inference_client

# --- Helper Functions (from your script) ---
def create_image_prompt(punjabi_text, sentiment_label, translator):
    """Creates the final prompt for the image generation model."""
    translated_text_obj = translator(punjabi_text, max_length=512)
    english_translation = translated_text_obj[0]['translation_text']
    
    prompt_prefix = f"cinematic photo of '{english_translation}'. "
    if sentiment_label == 'positive':
        prompt_suffix = "The scene is filled with warm, bright light, evoking a sense of joy, success, and optimism. A vibrant and uplifting atmosphere. professional color grading, high detail."
    elif sentiment_label == 'negative':
        prompt_suffix = "The scene is depicted with a moody, somber atmosphere, using a muted or cold color palette to convey struggle, sadness, or tension. dramatic lighting."
    else: # Neutral
        prompt_suffix = "The scene is objective and realistic, with natural lighting and a balanced composition, focusing on the literal description."

    full_prompt = prompt_prefix + prompt_suffix + " 8k, masterpiece, award-winning photography."
    negative_prompt = "low quality, blurry, ugly, deformed, noisy, poor details, bad anatomy, cartoon, anime, abstract, watermark, text, signature"
    
    return full_prompt, negative_prompt, english_translation

# --- Main App Interface ---
st.title("📊 Punjabi Sentiment-to-Image Generator")
st.write("UCS748")
st.write("Bhavya Gera")
st.write("102397001")
st.write("4Q17")

st.write(
    "This app analyzes the sentiment of a Punjabi sentence and generates an image "
    "that visually represents its meaning and emotion using AI."
)

# Load all the necessary models and clients
punjabi_sentiment_pipeline, id_to_label, translator, client = load_models()

if punjabi_sentiment_pipeline: # Only proceed if models loaded successfully
    # Create the user input form
    with st.form("generation_form"):
        punjabi_input = st.text_area("Enter a Punjabi sentence here:", "ਅਧਿਆਪਕ ਦੇਸ਼ ਤੇ ਕੌਮ ਦਾ ਨਿਰਮਾਤਾ ਹੁੰਦਾ ਹੈ।")
        generate_button = st.form_submit_button("Generate Image", type="primary")

    if generate_button and punjabi_input:
        with st.spinner('Analyzing sentiment...'):
            # 1. Get Sentiment
            result = punjabi_sentiment_pipeline(punjabi_input)[0]
            predicted_label_id = int(result['label'].split('_')[-1])
            sentiment_label = id_to_label[str(predicted_label_id)]
            st.success(f"**Predicted Sentiment:** {sentiment_label.capitalize()}")
        
        with st.spinner('Translating and creating prompt...'):
            # 2. Create Image Prompt
            image_prompt, negative_prompt, translated_text = create_image_prompt(punjabi_input, sentiment_label, translator)
            st.info(f"**Translated Text:** {translated_text}")
            with st.expander("See the full prompt sent to the AI"):
                st.write(f"**Prompt:** {image_prompt}")
                st.write(f"**Negative Prompt:** {negative_prompt}")
        
        with st.spinner('Generating image with SDXL... This may take a moment.'):
            # 3. Generate Image using the API
            try:
                generated_image = client.text_to_image(
                    image_prompt,
                    negative_prompt=negative_prompt,
                    model="stabilityai/stable-diffusion-xl-base-1.0",
                )
                
                st.image(generated_image, caption="Generated Image", use_column_width=True)

            except Exception as e:
                st.error(f"An error occurred during image generation: {e}")
else:
    st.warning("Could not initialize the models or API client. Please check your setup and API token.")