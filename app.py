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

# --- Global statistics ---
training_stats = {
    "total_samples": 0,
    "accuracy": 0.0,
    "spam_prevalence": 0.0
}


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
    global vectorizer, model, training_stats

    print("Starting up, please wait while the model trains...", flush=True)

    # Ensure NLTK stopwords are available
    try:
        stopwords.words('english')
    except LookupError:
        print("Downloading NLTK stopwords...", flush=True)
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

    total_samples = len(df)
    spam_samples = df[df['label'] == 1].shape[0]
    spam_prevalence = (spam_samples / total_samples) * 100

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Feature Extraction using TF-IDF
    vectorizer = TfidfVectorizer()
    X_train_transformed = vectorizer.fit_transform(X_train)

    # 4. Model Training (Multinomial Naive Bayes)
    model = MultinomialNB()
    model.fit(X_train_transformed, y_train)

    # Compute evaluation metric
    X_test_transformed = vectorizer.transform(X_test)
    accuracy = model.score(X_test_transformed, y_test) * 100

    # Store stats
    training_stats["total_samples"] = total_samples
    training_stats["accuracy"] = round(accuracy, 1)
    training_stats["spam_prevalence"] = round(spam_prevalence, 1)

    print("\n[SUCCESS] SMS Spam Classifier Model Trained and Ready.")
    return True


# --- FLASK ROUTES ---

# Global error handler for 405 (Method Not Allowed)
@app.errorhandler(405)
def method_not_allowed(e):
    """Provides a helpful error message when the user uses the wrong HTTP method."""
    print(f"\n[405 ERROR] Method {request.method} was attempted on path {request.path}.")
    return jsonify({
        "error": "405 Method Not Allowed",
        "description": "You tried to access a URL with the wrong HTTP method. Please ensure you are viewing the dashboard at http://127.0.0.1:5001/ and using the form to send data (which correctly sends a POST request to the /predict endpoint)."
    }), 405


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
        from flask import make_response
        response = make_response(f(*args, **kwargs))
        # Allows requests from any origin
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        # Explicitly allow all methods used by the route and preflight checks
        response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS, GET'
        return response

    return decorated_function


# Route to handle stats
@app.route('/stats', methods=['GET', 'OPTIONS'])
@cors_allow_all
def get_stats():
    if request.method == 'OPTIONS':
        return jsonify({}), 200
    if not model or not vectorizer:
        return jsonify({"error": "Model not trained. Check server startup logs."}), 500
    return jsonify(training_stats)


# Route to handle prediction requests (POST method required)
@app.route('/predict', methods=['POST', 'OPTIONS', 'GET'])
@cors_allow_all
def predict():
    # If the request is a GET (e.g., from typing the URL in the browser), redirect to the homepage.
    if request.method == 'GET':
        print("GET request received at /predict. Redirecting to dashboard.")
        return redirect('/')

    # Handle CORS preflight request
    if request.method == 'OPTIONS':
        return jsonify({}), 200

    # --- Start of POST Request Logic ---
    print(f"\n[DEBUG] Receiving POST request on /predict...")

    if not model or not vectorizer:
        print("[DEBUG] Model not ready.")
        return jsonify({"error": "Model not trained. Check server startup logs."}), 500

    data = request.get_json(silent=True)

    if data is None:
        print("[DEBUG] Failed to get JSON data. Check Content-Type header on client side.")
        return jsonify({"error": "Invalid or missing JSON data in request body."}), 400

    message = data.get('message', '')
    print(f"[DEBUG] Received message: '{message}'")

    if not message:
        return jsonify({"prediction": "No message provided."}), 400

    try:
        # Preprocess
        processed_message = preprocess_text(message)

        # Transform (using the fitted vectorizer)
        transformed_message = vectorizer.transform([processed_message])

        # Predict (Use probabilities with a lowered threshold for higher sensitivity to SPAM)
        prediction_probs = model.predict_proba(transformed_message)[0]
        spam_prob = prediction_probs[1]
        
        # Add basic fraud keywords to boost sensitivity when the model is uncertain
        fraud_keywords = ['fraud', 'scam', 'spam', 'win', 'prize', 'cash', 'lottery', 'urgent', 'winner']
        has_fraud_keywords = any(word in processed_message for word in fraud_keywords)
        
        if spam_prob > 0.15 or has_fraud_keywords:
            prediction_result = 1
        else:
            prediction_result = 0

        # Return label
        label = "SPAM 🚫" if prediction_result == 1 else "HAM (Legitimate) ✅"

        print(f"[DEBUG] Classification result: {'SPAM' if prediction_result == 1 else 'HAM'}")

        return jsonify({
            "message": message,
            "prediction": label,
            "success": True
        })
    except Exception as e:
        print(f"[ERROR] Prediction error: {e}")
        return jsonify({"error": "An internal error occurred during prediction."}), 500


if __name__ == '__main__':
    # Train the model before starting the server
    if load_model_and_vectorizer():
        print(f"\n* SMS Classifier is ready.")
        print(f"* Access the dashboard at: http://127.0.0.1:{PORT}/")
        app.run(debug=True, port=PORT)