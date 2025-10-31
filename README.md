# Punjabi Sentiment-Driven Image Generator 🎨

This project is a multimodal AI application that analyzes the sentiment of Punjabi text (a low-resource language) and generates a corresponding image using state-of-the-art models. The goal is to create visuals that capture not just the literal meaning of the text, but its emotional tone as well.

The application is built with a user-friendly web interface using Streamlit.

## 🌟 Demo

*(Here you would add a screenshot of your running Streamlit application. It makes the README much more engaging!)*

![App Screenshot](path/to/your/screenshot.png)

## ✨ Features

-   **Punjabi Sentiment Analysis:** Utilizes a fine-tuned `XLM-RoBERTa` model to classify Punjabi text into positive, negative, or neutral sentiment.
-   **Text Translation:** Translates the input Punjabi text into English to make it compatible with modern image generation models.
-   **Sentiment-Conditioned Prompt Engineering:** Intelligently combines the translated text with descriptive words based on the predicted sentiment to create a rich, emotionally-aware prompt.
-   **High-Quality Image Generation:** Leverages the powerful `Stable Diffusion XL (SDXL)` model via the Hugging Face Inference API to generate stunning, high-resolution images.
-   **Interactive Web UI:** A simple and intuitive web interface built with Streamlit allows anyone to easily use the application.

## ⚙️ Project Architecture

The application follows a sequential pipeline to process the user's input and generate an image:

1.  **User Input:** The user enters a Punjabi sentence into the Streamlit web interface.
2.  **Sentiment Analysis (Local):** The fine-tuned sentiment classification model (running locally) predicts the sentiment of the text.
3.  **Translation (Local):** The Helsinki-NLP model translates the Punjabi text to English.
4.  **Prompt Engineering:** A custom function constructs a detailed prompt by merging the translated text with stylistic and emotional keywords derived from the sentiment.
5.  **API Call:** The final prompt is sent to the Hugging Face Inference API, which hosts the SDXL model.
6.  **Image Generation (Cloud):** The SDXL model generates the image on Hugging Face's powerful infrastructure.
7.  **Display:** The generated image is sent back and displayed to the user in the Streamlit app.

## 🚀 Setup and Installation

Follow these steps to set up and run the project locally.

### Prerequisites

-   [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
-   Python 3.10
-   Git

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/punjabi_genai_project.git
cd punjabi_genai_project
2. Create and Activate Conda Environment
Create a dedicated Conda environment for this project to manage dependencies.
code
Bash
conda create --name genai_new python=3.10 -y
conda activate genai_new
3. Install Dependencies
Install all the required Python libraries from the requirements.txt file.
code
Bash
pip install -r requirements.txt
(If you don't have a requirements.txt file, create one with the following content:)
code
Code
pandas
scikit-learn
torch
transformers
sentencepiece
huggingface_hub
streamlit
Pillow
4. Set Up Hugging Face API Token
This project uses the Hugging Face Inference API for image generation. You need a free API token.
Get your token from your Hugging Face profile: Settings -> Access Tokens.
Set it as an environment variable. In your terminal, run:
code
Bash
# On Windows
set HF_TOKEN=hf_YourTokenGoesHere

# On macOS/Linux
export HF_TOKEN=hf_YourTokenGoesHere
Note: This variable is temporary and needs to be set every time you open a new terminal. For a permanent solution, add it to your system's environment variables.
🛠️ Usage
1. (Optional) Train the Sentiment Model
This repository should include a pre-trained sentiment model in the models/sentiment_classifier directory. However, if you want to re-train it on your own data:
Place your labeled CSV file (with sentence and sentiment columns) in the data/ directory.
Run the training script:
code
Bash
python scripts/01_train_sentiment_model.py
2. Run the Streamlit Application
This is the main way to use the project.
Ensure your genai_new environment is active.
Set your HF_TOKEN environment variable as described above.
Run the following command in your terminal:
code
Bash
streamlit run app.py
Your web browser will automatically open a new tab with the application running.
💻 Technologies Used
Backend & Machine Learning:
Python
PyTorch
Hugging Face Transformers (for local NLP pipelines)
Hugging Face Hub (for API client)
Frontend:
Streamlit
Models & APIs:
Sentiment: Fine-tuned xlm-roberta-base
Translation: Helsinki-NLP/opus-mt-pa-en
Image Generation: stabilityai/stable-diffusion-xl-base-1.0 via Hugging Face Inference API
🔮 Future Work
Expand Dataset: Add more diverse data, especially with negative and neutral sentiments, to improve the sentiment model's robustness.
Compare Image Models: Integrate options to use other models like DALL-E 3 or Midjourney via their respective APIs.
Improve Cultural Nuance: Experiment with fine-tuning image models on culturally relevant Punjabi imagery to reduce Western bias.
User Feedback Loop: Add a feature in the app for users to rate the generated images, which can be used to collect data for improving the prompt engineering.
📄 License
This project is licensed under the MIT License. See the LICENSE file for details.