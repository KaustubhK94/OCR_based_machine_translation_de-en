import os
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from transformers import MarianTokenizer, MarianMTModel
from PIL import Image
import re
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
import re

app = Flask(__name__, template_folder='OCR_Machine_translation/templates')

# Define the path for uploading and serving images
UPLOAD_FOLDER = 'OCR_Machine_translation/static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the translation model and tokenizer
tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-de-en")
model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-de-en")

# Function to translate text
def translate_text(input_text):
    inputs = tokenizer(input_text, return_tensors="pt")
    translation = model.generate(**inputs)
    translated_text = tokenizer.batch_decode(translation, skip_special_tokens=True)
     # Tokenize into sentences and join with HTML line breaks
    translated_text = "\n".join(translated_text)
    translated_text = translated_text.replace(". ", "\n")
    return translated_text

# Function to perform OCR on an image
def perform_ocr(image_path):
    # Open the image using PIL
    image = Image.open(image_path)

    # Perform OCR using pytesseract
    extracted_text = pytesseract.image_to_string(image, lang='deu')

    return extracted_text

@app.route("/", methods=["GET", "POST"])
def index():
    translated_text = ""

    if request.method == "POST":
        # Check if a file was uploaded
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']

        # If the user does not select a file, the browser will submit an empty file
        if file.filename == '':
            return redirect(request.url)

        if file:
            # Save the uploaded image to the upload folder
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)

            # Perform OCR on the uploaded image
            extracted_text = perform_ocr(filename)

            # Translate the extracted text and replace newlines with HTML line breaks
            translated_text = translate_text(extracted_text).replace("\n", "<br>")

    return render_template("index.html", translated_text=translated_text)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    app.run()
