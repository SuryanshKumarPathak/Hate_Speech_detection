from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pickle
import re
import nltk
import string

nltk.download('stopwords')

app = Flask(__name__, static_folder="static")
CORS(app)  # ✅ Allow frontend to talk to backend

# Load model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# Preprocessing (same as your Colab)
steamer = nltk.SnowballStemmer("english")
stopwords = set(nltk.corpus.stopwords.words("english"))

def clean(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = [word for word in text.split(' ') if word not in stopwords]
    text = " ".join(text)
    text = [steamer.stem(word) for word in text.split(' ')]
    text = " ".join(text)
    return text

@app.route('/')
def home():
    return send_from_directory('static', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text', '')

    cleaned_text = clean(text)
    vectorized = vectorizer.transform([cleaned_text]).toarray()
    prediction = model.predict(vectorized)[0]

    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
