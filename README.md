# 🌟 Punjabi Text-to-Image Generator (Gradio Version) 🎨

This project is a multimodal AI application that converts **Punjabi text into English**, generates a cinematic prompt, and produces a high-quality image using **Stable Diffusion XL (SDXL)** via the Hugging Face Inference API.

The system is built with a modern and interactive **Gradio web interface**, ensuring fast and intuitive user interaction.

---

## 🖼️ Demo Screenshot  
<img width="940" height="397" alt="image" src="https://github.com/user-attachments/assets/5c79d702-13f7-46e5-8238-0f00d5dc4816" />

---

## ✨ Features

- **Punjabi → English Translation**  
  Powered by the `Helsinki-NLP/opus-mt-pa-en` model.

- **Cinematic Prompt Generator**  
  Automatically creates visually rich prompts describing the translated text.

- **High-Quality Image Generation**  
  Uses **Stable Diffusion XL** (via Hugging Face Inference API).

- **Interactive Gradio UI**  
  Clean, fast, responsive — replaces the old Streamlit version.

- **No Sentiment Analysis**  
  The system now uses *only translation and prompt generation*, simplifying output.

---

## ⚙️ Project Architecture

1. **User Input** → Punjabi text  
2. **Translation** → Local translation to English  
3. **Prompt Creation** → Text converted into cinematic photography prompt  
4. **Inference API Call** → Sent to Hugging Face SDXL  
5. **Image Output** → Displayed instantly in Gradio UI  

---

## 🚀 Setup & Installation

### **Prerequisites**
- Python 3.10  
- Git  
- Hugging Face API Token  
- (Optional) Conda for isolated environments

---

```bash
# commands to run the system

conda create --name genai_new python=3.10 -y
conda activate genai_new

pip install -r requirements

pip install gradio transformers sentencepiece huggingface_hub Pillow torch pandas

set HF_TOKEN=hf_your_token_here

#Run the Gradio application:
python app.py
