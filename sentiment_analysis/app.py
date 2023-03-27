!pip install flask
from flask import Flask, render_template, request
import pickle
import re
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)

# Load the trained model and vectorizer
with open("model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

def preprocess_text(text):
    # Replace <br /> with a space
    text = re.sub(r"<br\s*/?>", " ", text)
    return text

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        text = request.form["text"]
        preprocessed_text = preprocess_text(text)
        features = vectorizer.transform([preprocessed_text])
        prediction = model.predict(features)[0]
        sentiment = "Positive" if prediction == 1 else "Negative"
        return render_template("index.html", sentiment=sentiment)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
