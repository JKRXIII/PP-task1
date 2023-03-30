from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# задаем модель и токенизатор
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# задаем конвейер для классификации текстовых данных
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

# пример текстовых данных для классификации
text = "This movie was not good enough. It might be better."

# классификация текста
result = classifier(text)

# вывод результата
print(f"Label: {result[0]['label']}, Score: {result[0]['score']}")
