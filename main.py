from fastapi import FastAPI
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

app = FastAPI()

# Load the model and tokenizer
model_name = "Hareesh123/model"  # Replace with your actual model name
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

@app.post("/classification/")
async def classify_text(text: str):
    result = classifier(text)
    return result
