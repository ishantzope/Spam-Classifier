from flask import Flask, request, jsonify, redirect
import pandas as pd
import string
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import nltk
import os
from functools import wraps

# --- Flask App Initialization ---
app = Flask(__name__)
# The port is changed to 5001
PORT = 5001

# --- Global variables for the trained model and vectorizer ---
vectorizer = None
model = None


# --- Helper Function (from spam_classifier.py) ---
def preprocess_text(text):
    """
    Cleans and processes the text: removes punctuation, tokenizes,
    removes stopwords, and returns a clean string.
    """
    if not isinstance(text, str):
        return ""

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Tokenize and remove stopwords
    try:
        stop_words = stopwords.words('english')
    except LookupError:
        stop_words = []

    words = [word.lower() for word in text.split() if word.lower() not in stop_words]

    return " ".join(words)


# --- Model Training/Loading (Run once on startup) ---
def load_model_and_vectorizer():
    global vectorizer, model

    # Ensure NLTK stopwords are available
    try:
        stopwords.words('english')
    except LookupError:
        print("Downloading NLTK stopwords...")
        nltk.download('stopwords')

    # 1. Load the Dataset
    file_path = 'spam.csv'
    if not os.path.exists(file_path):
        print(f"ERROR: Dataset '{file_path}' not found. Please place it in the same directory as app.py.")
        return False

    try:
        df = pd.read_csv(file_path, encoding='latin-1')
        df = df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], errors='ignore')
        df.columns = ['label', 'message']
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return False

    df['label'] = df['label'].map({'ham': 0, 'spam': 1})

    # 2. Apply Text Preprocessing
    df['processed_message'] = df['message'].apply(preprocess_text)

    X = df['processed_message']
    y = df['label']

    # Simple split (using a tiny test size as the model is for deployment)
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.01, random_state=42)

    # 3. Feature Extraction using TF-IDF
    vectorizer = TfidfVectorizer()
    X_train_transformed = vectorizer.fit_transform(X_train)

    # 4. Model Training (Multinomial Naive Bayes)
    model = MultinomialNB()
    model.fit(X_train_transformed, y_train)

    print("\n[SUCCESS] SMS Spam Classifier Model Trained and Ready.")
    return True


# --- FLASK ROUTES ---

# Route to serve the HTML dashboard directly from the server
@app.route('/', methods=['GET'])
def index():
    """Reads the content of index.html and serves it."""
    try:
        with open('index.html', 'r', encoding='utf-8') as f:
            html_content = f.read()
        return html_content
    except FileNotFoundError:
        return jsonify({"error": "index.html not found. Please ensure it is in the same directory as app.py"}), 500


# Enable CORS for development (allowing index.html to fetch from the server)
def cors_allow_all(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        response = f(*args, **kwargs)
        # Allows requests from any origin, which is required when index.html is loaded via file://
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
        return response

    return decorated_function


# Route to handle prediction requests (POST method required)
@app.route('/predict', methods=['POST', 'OPTIONS'])
@cors_allow_all
def predict():
    if request.method == 'OPTIONS':
        # Handle CORS preflight request, required by the browser before a POST request
        return jsonify({}), 200

    if not model or not vectorizer:
        return jsonify({"error": "Model not trained. Check server startup logs."}), 500

    data = request.get_json(silent=True)
    message = data.get('message', '')

    if not message:
        return jsonify({"prediction": "No message provided."}), 400

    try:
        # Preprocess
        processed_message = preprocess_text(message)

        # Transform (using the fitted vectorizer)
        transformed_message = vectorizer.transform([processed_message])

        # Predict
        prediction_result = model.predict(transformed_message)[0]

        # Return label
        label = "SPAM 🚫" if prediction_result == 1 else "HAM (Legitimate) ✅"

        return jsonify({
            "message": message,
            "prediction": label,
            "success": True
        })
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({"error": "An internal error occurred during prediction."}), 500


if __name__ == '__main__':
    # Train the model before starting the server
    if load_model_and_vectorizer():
        print(f"\n* SMS Classifier is ready.")
        print(f"* Access the dashboard at: http://127.0.0.1:{PORT}/")
        app.run(debug=True, port=PORT)