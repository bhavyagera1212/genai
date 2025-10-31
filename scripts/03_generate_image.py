# scripts/02_generate_image.py

import torch
import os
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from PIL import Image

# --- NEW: Import the Hugging Face API Client ---
from huggingface_hub import InferenceClient

# --- Configuration ---
SENTIMENT_MODEL_PATH = 'models/sentiment_classifier'
TRANSLATION_MODEL_NAME = "Helsinki-NLP/opus-mt-pa-en"

# --- NEW: Set up the Inference Client ---
# Make sure to set your token as an environment variable
# In your terminal: set HF_TOKEN=hf_your_token_here
HF_API_TOKEN = os.getenv("HF_TOKEN")
if not HF_API_TOKEN:
    raise ValueError("Missing Hugging Face API token. Please set the HF_TOKEN environment variable.")

# Initialize the client WITHOUT the provider argument for standard API usage
try:
    client = InferenceClient(token=HF_API_TOKEN)
    print("Hugging Face Inference Client initialized successfully.")
except Exception as e:
    print(f"Failed to initialize Inference Client: {e}")
    exit()

# --- 1. Load Trained Sentiment Model and Translator (runs locally) ---
print(f"Loading local NLP models...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device set for NLP models: {device}")
try:
    # Load sentiment pipeline
    sentiment_tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL_PATH)
    sentiment_model = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL_PATH)
    punjabi_sentiment_pipeline = pipeline(
        "sentiment-analysis", model=sentiment_model, tokenizer=sentiment_tokenizer,
        device=0 if device == "cuda" else -1
    )
    with open(f"{SENTIMENT_MODEL_PATH}/id_to_label.json", 'r', encoding='utf-8') as f:
        id_to_label = json.load(f)

    # Load translation pipeline
    translator = pipeline(
        "translation", model=TRANSLATION_MODEL_NAME,
        device=0 if device == "cuda" else -1
    )
    print("Local NLP models loaded successfully.")
except Exception as e:
    print(f"ERROR: Could not load local NLP models. {e}")
    exit()

# --- Function to Get Sentiment ---
def get_sentiment(text: str) -> str:
    """Predicts sentiment of Punjabi text using the trained model."""
    result = punjabi_sentiment_pipeline(text)[0]
    predicted_label_id = int(result['label'].split('_')[-1])
    return id_to_label[str(predicted_label_id)]

# --- Function for Prompt Engineering (Enhanced for SDXL) ---
def create_image_prompt(punjabi_text: str, sentiment_label: str) -> str:
    """
    Translates Punjabi text and predicted sentiment into an effective English prompt
    for the SDXL image generation model.
    """
    translated_text_obj = translator(punjabi_text, max_length=512)
    english_translation = translated_text_obj[0]['translation_text']
    print(f"Translated Text: {english_translation}")

    prompt_prefix = f"cinematic photo of '{english_translation}'. "
    if sentiment_label == 'positive':
        prompt_suffix = "The scene is filled with warm, bright light, evoking a sense of joy, success, and optimism. A vibrant and uplifting atmosphere. professional color grading, high detail."
    elif sentiment_label == 'negative':
        prompt_suffix = "The scene is depicted with a moody, somber atmosphere, using a muted or cold color palette to convey struggle, sadness, or tension. dramatic lighting."
    else: # Neutral
        prompt_suffix = "The scene is objective and realistic, with natural lighting and a balanced composition, focusing on the literal description."

    full_prompt = prompt_prefix + prompt_suffix + " 8k, masterpiece, award-winning photography."
    negative_prompt = "low quality, blurry, ugly, deformed, noisy, poor details, bad anatomy, cartoon, anime, abstract, watermark, text, signature"
    
    return full_prompt, negative_prompt

# --- NEW: Image Generation Function using InferenceClient ---
def generate_image(prompt: str, negative_prompt: str, output_path: str = "generated_image.png"):
    """
    Generates an image using the Hugging Face Inference API via InferenceClient.
    """
    print(f"Sending prompt to Hugging Face API via InferenceClient: {prompt}")
    try:
        # The client handles the API call
        image = client.text_to_image(
            prompt,
            negative_prompt=negative_prompt,
            model="stabilityai/stable-diffusion-xl-base-1.0",
            # You can add other parameters here if needed, e.g., height=1024, width=1024
        )
        # The output 'image' is a PIL.Image object, so we just save it.
        image.save(output_path)
        print(f"Image saved to {output_path}")
        return output_path
    except Exception as e:
        print(f"An error occurred during image generation with the API: {e}")
        return None

# --- Main Execution Flow (no changes here) ---
def main():
    punjabi_statement = input("Enter a Punjabi statement to generate an image: ")
    sentiment_label = get_sentiment(punjabi_statement)
    print(f"Predicted Sentiment: {sentiment_label}")
    image_prompt, negative_prompt = create_image_prompt(punjabi_statement, sentiment_label)
    print(f"Final English Prompt: {image_prompt}")
    output_filename = f"generated_image_API_{sentiment_label}.png"
    generated_image_path = generate_image(image_prompt, negative_prompt, output_filename)

    if generated_image_path:
        print(f"\nImage generation complete! Check '{generated_image_path}'")
    else:
        print("\nImage generation failed.")

if __name__ == "__main__":
    main()