# -*- coding: utf-8 -*-
"""
@author: Santosh Kumar
"""
import os
import time
from flask import Flask, request, jsonify, render_template
import requests
from bs4 import BeautifulSoup
import spacy
import nltk
from nltk.tokenize import sent_tokenize
import openai
from dotenv import load_dotenv

load_dotenv(dotenv_path='E:\AIML\WEB QnA\.env')

# Initialize Flask app
app = Flask(__name__)

# Load SpaCy NLP model
nlp = spacy.load("en_core_web_sm")

# OpenAI API Key 
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in .env file.")

# Function to extract text content from a URL
def extract_content_from_url(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")
        # Extract all text from the page
        text = soup.get_text(separator=" ")
        return text.strip()
    except Exception as e:
        return str(e)

# Function to summarize or clean extracted content
def preprocess_content(content):
    # Tokenize into sentences using NLTK
    sentences = sent_tokenize(content)
    # Limit to the first 50 sentences for brevity
    return " ".join(sentences[:50])

# Function to answer questions using OpenAI GPT model
def get_answer_from_openai(question, context):
    try:
        prompt = f"Context: {context}\n\nQuestion: {question}\nAnswer:"
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=150,
            temperature=0.7,
        )
        return response.choices[0].text.strip()
    except Exception as e:
        return str(e)
# Routes
@app.route('/')
def index():
    return render_template('index.xml')

@app.route('/ingest', methods=['POST'])
def ingest_urls():
    data = request.json
    urls = data.get("urls", [])
    
    if not urls:
        return jsonify({"error": "No URLs provided"}), 400

    extracted_content = {}
    for url in urls:
        content = extract_content_from_url(url)
        processed_content = preprocess_content(content)
        extracted_content[url] = processed_content

    return jsonify({"extracted_content": extracted_content})

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.json
    question = data.get("question", "")
    context = data.get("context", "")

    if not question or not context:
        return jsonify({"error": "Question or context missing"}), 400

    answer = get_answer_from_openai(question, context)
    return jsonify({"answer": answer})

# Proceed with file handling
def on_any_event(event):
    if event.event_type == 'created':
        time.sleep(0.5)

# Run the Flask app
if __name__ == '__main__':   
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)  


