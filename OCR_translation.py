import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
import re
print("Available languages:", pytesseract.get_languages(config=""))

# You should see 'deu' (German) in the list of available languages
import pytesseract
from PIL import Image
import matplotlib.pyplot as plt


image = Image.open('/content/drive/MyDrive/Colab Notebooks/page19.jpg')

extracted_text = pytesseract.image_to_string(image, lang='deu')

plt.imshow(image)

print(extracted_text)

from transformers import MarianTokenizer, MarianMTModel

# Load the tokenizer and model
tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-de-en")
model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-de-en")

# Tokenize the text
inputs = tokenizer(extracted_text, return_tensors="pt")

# Translate from German to English
translation = model.generate(**inputs)

# Decode and print the translation
translated_text = tokenizer.batch_decode(translation, skip_special_tokens=True)

translated_text = "\n".join(translated_text)
print(translated_text)

translated_text = translated_text.replace(". ", "\n")
print(translated_text)

