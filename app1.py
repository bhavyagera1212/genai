import gradio as gr
import torch
import os
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from huggingface_hub import InferenceClient


# -----------------------------
# Load Models (cached)
# -----------------------------
sentiment_pipeline = None
id_to_label = None
translator = None
client = None


def load_all_models():
    global sentiment_pipeline, id_to_label, translator, client

    if sentiment_pipeline is not None:
        return

    print("Loading models...")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Sentiment model
    sentiment_model_path = "models/sentiment_classifier"
    tokenizer = AutoTokenizer.from_pretrained(sentiment_model_path)
    model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_path)

    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model=model,
        tokenizer=tokenizer,
        device=0 if device == "cuda" else -1
    )

    with open(f"{sentiment_model_path}/id_to_label.json", "r", encoding="utf-8") as f:
        id_to_label = json.load(f)

    # Punjabi → English translation
    translator = pipeline(
        "translation",
        model="Helsinki-NLP/opus-mt-pa-en",
        device=0 if device == "cuda" else -1
    )

    # HF API Client
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise gr.Error("HF_TOKEN environment variable is missing!")

    client = InferenceClient(token=hf_token)


load_all_models()


# -----------------------------
# Helper Function
# -----------------------------
def create_prompt(punjabi_text, sentiment):
    translated_text_obj = translator(punjabi_text, max_length=512)
    english = translated_text_obj[0]["translation_text"]

    prefix = f"cinematic photo of '{english}'. "

    if sentiment == "positive":
        suffix = (
            "Warm bright lighting, joyful, optimistic atmosphere, vibrant colors, "
            "professional color grading, high detail."
        )
    elif sentiment == "negative":
        suffix = (
            "Moody dark lighting, somber emotional tone, muted colors, dramatic shadows."
        )
    else:
        suffix = (
            "Neutral realistic lighting, objective atmosphere, natural colors."
        )

    full_prompt = prefix + suffix + " 8k, masterpiece, award-winning photography."

    negative_prompt = (
        "low quality, blurry, deformed, noisy, bad anatomy, cartoon, watermark, text"
    )

    return full_prompt, negative_prompt, english


# -----------------------------
# Main Function for App
# -----------------------------
def generate_output(punjabi_sentence):

    # 1. Predict sentiment
    result = sentiment_pipeline(punjabi_sentence)[0]
    pred_id = int(result["label"].split("_")[-1])
    sentiment = id_to_label[str(pred_id)]

    # 2. Create prompt
    prompt, neg_prompt, eng = create_prompt(punjabi_sentence, sentiment)

    # 3. Generate image through HF API
    try:
        image = client.text_to_image(
            prompt,
            negative_prompt=neg_prompt,
            model="stabilityai/stable-diffusion-xl-base-1.0"
        )
    except Exception as e:
        raise gr.Error(f"Image generation failed: {e}")

    return (
        sentiment.capitalize(),
        eng,
        prompt,
        image
    )


# -----------------------------
# Gradio UI
# -----------------------------
with gr.Blocks(title="Punjabi Sentiment-to-Image Generator") as demo:

    gr.Markdown("""
    # 📊 Punjabi Sentiment-to-Image Generator  
    **UCS748**  
    **Bhavya Gera — 102397001 — 4Q17**  
    Analyze Punjabi text → detect sentiment → generate an AI image.
    """)

    input_box = gr.Textbox(
        label="Enter Punjabi Sentence",
        value="ਅਧਿਆਪਕ ਦੇਸ਼ ਤੇ ਕੌਮ ਦਾ ਨਿਰਮਾਤਾ ਹੁੰਦਾ ਹੈ।",
        lines=3
    )

    btn = gr.Button("Generate Image")

    sentiment_out = gr.Textbox(label="Predicted Sentiment")
    translated_out = gr.Textbox(label="English Translation")
    prompt_out = gr.Textbox(label="Final Prompt Sent to AI", lines=5)
    image_out = gr.Image(label="Generated Image")

    btn.click(
        generate_output,
        inputs=input_box,
        outputs=[sentiment_out, translated_out, prompt_out, image_out]
    )


demo.launch()
