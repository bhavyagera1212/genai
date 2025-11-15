# app_gradio.py

import os
import json
import io
import torch

import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from huggingface_hub import InferenceClient
from PIL import Image
from functools import lru_cache


# ---------- Model & Client Loading (Cached) ----------

@lru_cache(maxsize=1)
def load_models():
    """
    Loads and caches the NLP models and the Inference Client.
    Called only once thanks to lru_cache.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- 1. Load Trained Sentiment Model (local) ---
    sentiment_model_path = "models/sentiment_classifier"
    sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_path)
    sentiment_model = AutoModelForSequenceClassification.from_pretrained(
        sentiment_model_path
    )

    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model=sentiment_model,
        tokenizer=sentiment_tokenizer,
        device=0 if device == "cuda" else -1,
    )

    with open(f"{sentiment_model_path}/id_to_label.json", "r", encoding="utf-8") as f:
        id_to_label = json.load(f)

    # --- 2. Translation pipeline (Punjabi -> English) ---
    translation_pipeline = pipeline(
        "translation",
        model="Helsinki-NLP/opus-mt-pa-en",
        device=0 if device == "cuda" else -1,
    )

    # --- 3. Hugging Face Inference Client for SDXL ---
    hf_api_token = os.getenv("HF_TOKEN")
    if not hf_api_token:
        # Return None for client so we can raise a Gradio error later
        inference_client = None
    else:
        inference_client = InferenceClient(token=hf_api_token)

    return sentiment_pipeline, id_to_label, translation_pipeline, inference_client


# ---------- Helper Function ----------

def create_image_prompt(punjabi_text, sentiment_label, translator):
    """
    Creates the final prompt for the image generation model.
    """
    translated_text_obj = translator(punjabi_text, max_length=512)
    english_translation = translated_text_obj[0]["translation_text"]

    prompt_prefix = f"cinematic photo of '{english_translation}'. "

    if sentiment_label == "positive":
        prompt_suffix = (
            "The scene is filled with warm, bright light, evoking a sense of joy, "
            "success, and optimism. A vibrant and uplifting atmosphere. "
            "professional color grading, high detail."
        )
    elif sentiment_label == "negative":
        prompt_suffix = (
            "The scene is depicted with a moody, somber atmosphere, using a muted "
            "or cold color palette to convey struggle, sadness, or tension. "
            "dramatic lighting."
        )
    else:  # Neutral
        prompt_suffix = (
            "The scene is objective and realistic, with natural lighting and a "
            "balanced composition, focusing on the literal description."
        )

    full_prompt = (
        prompt_prefix
        + prompt_suffix
        + " 8k, masterpiece, award-winning photography."
    )
    negative_prompt = (
        "low quality, blurry, ugly, deformed, noisy, poor details, bad anatomy, "
        "cartoon, anime, abstract, watermark, text, signature"
    )

    return full_prompt, negative_prompt, english_translation


# ---------- Main Inference Function for Gradio ----------

def punjabi_sentiment_to_image(punjabi_text: str):
    """
    Main function that:
      1. Runs sentiment analysis
      2. Translates Punjabi -> English
      3. Builds SDXL prompt
      4. Calls HF Inference endpoint to generate image
    Returns:
      sentiment_label, translated_text, full_prompt, negative_prompt, image(PIL)
    """
    if not punjabi_text or not punjabi_text.strip():
        raise gr.Error("Please enter a Punjabi sentence.")

    sentiment_pipeline, id_to_label, translator, client = load_models()

    if client is None:
        raise gr.Error(
            "Hugging Face API token (HF_TOKEN) not found in environment variables."
        )

    # 1. Get Sentiment
    result = sentiment_pipeline(punjabi_text)[0]
    # Assuming label format is something like "LABEL_0", "LABEL_1", etc.
    predicted_label_id = int(result["label"].split("_")[-1])
    sentiment_label = id_to_label[str(predicted_label_id)]

    # 2. Create prompt (also performs translation)
    image_prompt, negative_prompt, translated_text = create_image_prompt(
        punjabi_text, sentiment_label, translator
    )

    # 3. Call text-to-image API (SDXL)
    try:
        img_bytes = client.text_to_image(
            image_prompt,
            negative_prompt=negative_prompt,
            model="stabilityai/stable-diffusion-xl-base-1.0",
        )
        # InferenceClient returns raw bytes; convert to PIL Image
        if isinstance(img_bytes, (bytes, bytearray)):
            image = Image.open(io.BytesIO(img_bytes))
        else:
            # If client ever returns a PIL.Image directly
            image = img_bytes
    except Exception as e:
        raise gr.Error(f"Error during image generation: {e}")

    # Nicely formatted sentiment label
    sentiment_display = sentiment_label.capitalize()

    return (
        sentiment_display,
        translated_text,
        image_prompt,
        negative_prompt,
        image,
    )


# ---------- Gradio UI ----------

with gr.Blocks(title="Punjabi Sentiment-to-Image Generator") as demo:
    gr.Markdown(
        """
# 🎨 Punjabi Sentiment-to-Image Generator

Enter a *Punjabi sentence* and this app will:

1. Detect the *sentiment* using your fine-tuned classifier  
2. Translate it to *English*  
3. Create a cinematic *image prompt* based on the sentiment  
4. Generate an image with *Stable Diffusion XL* 🚀
"""
    )

    with gr.Row():
        with gr.Column(scale=1):
            punjabi_input = gr.Textbox(
                label="Punjabi Sentence",
                value="ਅਧਿਆਪਕ ਦੇਸ਼ ਤੇ ਕੌਮ ਦਾ ਨਿਰਮਾਤਾ ਹੁੰਦਾ ਹੈ।",
                lines=4,
                placeholder="ਇੱਥੇ ਪੰਜਾਬੀ ਵਿੱਚ ਆਪਣਾ ਵਾਕ ਲਿਖੋ...",
            )
            generate_btn = gr.Button("Generate Image", variant="primary")

        with gr.Column(scale=1):
            sentiment_out = gr.Textbox(
                label="Predicted Sentiment", interactive=False
            )
            translation_out = gr.Textbox(
                label="Translated Text (Punjabi → English)", interactive=False
            )

    with gr.Accordion("Prompt Details (for SDXL)", open=False):
        prompt_out = gr.Textbox(
            label="Final Prompt", lines=4, interactive=False
        )
        negative_prompt_out = gr.Textbox(
            label="Negative Prompt", lines=3, interactive=False
        )

    image_out = gr.Image(label="Generated Image", type="pil")

    generate_btn.click(
        fn= Punjabi_sentiment_to_image := punjabi_sentiment_to_image,
        inputs=punjabi_input,
        outputs=[
            sentiment_out,
            translation_out,
            prompt_out,
            negative_prompt_out,
            image_out,
        ],
    )

if __name__ == "__main__":
    demo.launch()