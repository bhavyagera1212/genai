# scripts/02_generate_image.py

import torch
import os
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from diffusers import StableDiffusionPipeline
from PIL import Image

# --- Configuration ---
SENTIMENT_MODEL_PATH = 'models/sentiment_classifier'
TRANSLATION_MODEL_NAME = "Helsinki-NLP/opus-mt-pa-en" # Punjabi to English translation
IMAGE_GENERATION_MODEL_ID = "runwayml/stable-diffusion-v1-5" # A widely used Stable Diffusion model

# --- 1. Load Trained Sentiment Model ---
print(f"Loading sentiment model from {SENTIMENT_MODEL_PATH}...")
try:
    sentiment_tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL_PATH)
    sentiment_model = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL_PATH)
    punjabi_sentiment_pipeline = pipeline("sentiment-analysis", model=sentiment_model, tokenizer=sentiment_tokenizer)

    with open(f"{SENTIMENT_MODEL_PATH}/id_to_label.json", 'r', encoding='utf-8') as f:
        id_to_label = json.load(f)
    print("Sentiment model loaded successfully.")
except Exception as e:
    print(f"ERROR: Could not load sentiment model. Please ensure you've run 01_train_sentiment_model.py successfully. {e}")
    exit()

# --- 2. Load Translation Model ---
print(f"Loading Punjabi-to-English translation model '{TRANSLATION_MODEL_NAME}'...")
try:
    translator = pipeline("translation", model=TRANSLATION_MODEL_NAME)
    print("Translation model loaded successfully.")
except Exception as e:
    print(f"ERROR: Could not load translation model. Please check internet connection or model name. {e}")
    exit()

# --- 3. Load Image Generation Model ---
print(f"Loading image generation model '{IMAGE_GENERATION_MODEL_ID}'...")
try:
    # Use GPU if available, otherwise CPU (GPU is highly recommended for speed)
    if torch.cuda.is_available():
        image_pipeline = StableDiffusionPipeline.from_pretrained(IMAGE_GENERATION_MODEL_ID, torch_dtype=torch.float16)
        image_pipeline.to("cuda")
        print("Stable Diffusion pipeline loaded to GPU.")
    else:
        image_pipeline = StableDiffusionPipeline.from_pretrained(IMAGE_GENERATION_MODEL_ID)
        print("Stable Diffusion pipeline loaded to CPU (this will be slow).")
except Exception as e:
    print(f"ERROR: Could not load Stable Diffusion model. Please check 'diffusers' installation and internet connection. {e}")
    exit()

# --- Function to Get Sentiment ---
def get_sentiment(text: str) -> str:
    """Predicts sentiment of Punjabi text using the trained model."""
    result = punjabi_sentiment_pipeline(text)[0]
    # The pipeline returns label IDs, we need to map them back to string labels
    predicted_label_id = int(result['label'].split('_')[-1]) # e.g., 'LABEL_0' -> 0
    return id_to_label[str(predicted_label_id)] # Access with string key if JSON loaded as string keys

# --- Function for Prompt Engineering ---
def create_image_prompt(punjabi_text: str, sentiment_label: str) -> str:
    """
    Translates Punjabi text and predicted sentiment into an effective English prompt
    for an image generation model.
    """
    # Translate the Punjabi text to English
    translated_text_obj = translator(punjabi_text, max_length=512)
    english_translation = translated_text_obj[0]['translation_text']
    print(f"Translated Text: {english_translation}")

    # Craft the prompt based on sentiment
    prompt_prefix = f"A high-quality, professional photograph, realistic style, detailed, showing '{english_translation}'. "

    if sentiment_label == 'positive':
        prompt_suffix = "The scene should be bright, vibrant, and convey feelings of joy, hope, success, and positive energy. Beautiful lighting, happy atmosphere."
    elif sentiment_label == 'negative':
        # Adjust if you have 'negative' examples; current data is all positive
        prompt_suffix = "The scene should convey feelings of sadness, struggle, or challenge. Muted colors, somber atmosphere, a sense of overcoming difficulty."
    else: # Neutral or unhandled sentiment
        prompt_suffix = "The scene should be objective and descriptive, focusing on the literal elements of the translated text. Natural lighting, balanced composition."

    full_prompt = prompt_prefix + prompt_suffix + ", 4k, trending on ArtStation, cinematic."
    
    # Optional: Add negative prompt to improve image quality
    negative_prompt = "low quality, blurry, ugly, deformed, noisy, poor details, bad anatomy, grayscale, abstract, cartoon"
    
    return full_prompt, negative_prompt

# --- Function for Image Generation ---
def generate_image(prompt: str, negative_prompt: str, output_path: str = "generated_image.png"):
    """
    Generates an image using the Stable Diffusion pipeline.
    """
    print(f"Generating image with prompt: {prompt}")
    print(f"Using negative prompt: {negative_prompt}")
    
    # You can adjust num_inference_steps for quality vs speed.
    # Higher steps = better quality but slower. 20-30 is a good balance.
    image = image_pipeline(prompt, negative_prompt=negative_prompt, num_inference_steps=30).images[0]
    image.save(output_path)
    print(f"Image saved to {output_path}")
    return output_path

# --- Main Execution Flow ---
def main():
    punjabi_statement = input("Enter a Punjabi statement to generate an image: ")

    # 1. Get Sentiment
    sentiment_label = get_sentiment(punjabi_statement)
    print(f"Predicted Sentiment: {sentiment_label}")

    # 2. Create Image Prompt
    image_prompt, negative_prompt = create_image_prompt(punjabi_statement, sentiment_label)
    print(f"Final English Prompt: {image_prompt}")

    # 3. Generate Image
    output_filename = f"generated_image_{sentiment_label}_{punjabi_statement[:10].replace(' ', '_')}.png"
    generated_image_path = generate_image(image_prompt, negative_prompt, output_filename)

    if generated_image_path:
        print(f"\nImage generation complete! Check '{generated_image_path}'")
    else:
        print("\nImage generation failed.")

if __name__ == "__main__":
    main()