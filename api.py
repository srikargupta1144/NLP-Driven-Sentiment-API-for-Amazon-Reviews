from flask import Flask, request, jsonify, send_file, render_template
import re
from io import BytesIO
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import matplotlib.pyplot as plt
import pickle
import base64 
import nltk
import logging

# Set matplotlib backend to Agg
import matplotlib
matplotlib.use('Agg')

nltk.download('stopwords')
STOPWORDS = set(stopwords.words("english"))

# Initialize Flask app
app = Flask(__name__)

# Load models once at the start
with open("Models/xgb.pkl", "rb") as model_file:
    predictor = pickle.load(model_file)
with open("Models/scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)
with open("Models/CountVectorizer.pkl", "rb") as cv_file:
    cv = pickle.load(cv_file)

# Set up logging
logging.basicConfig(level=logging.INFO)

@app.route("/test", methods=["GET"])
def test():
    return "Test request received successfully. Service is running."

@app.route("/", methods=["GET", "POST"])
def home():
    return render_template("landing.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" in request.files:
            file = request.files["file"]
            data = pd.read_csv(file)
            predictions, graph = bulk_prediction(data)
            
            response = send_file(
                predictions, 
                mimetype="text/csv", 
                as_attachment=True, 
                download_name="Predictions.csv"
            )
            response.headers["X-Graph-Exists"] = "true"
            response.headers["X-Graph-Data"] = base64.b64encode(graph.getbuffer()).decode("ascii")
            return response
        
        elif "text" in request.json:
            text_input = request.json['text']
            predicted_sentiment = single_prediction(text_input)
            return jsonify({"prediction": predicted_sentiment})
        else:
            return jsonify({"error": "No valid input provided"}), 400
    except Exception as e:
        logging.error(f"Error in prediction: {str(e)}")
        return jsonify({"error": str(e)}), 500

def preprocess_text(text):
    """Preprocess text for prediction."""
    stemmer = PorterStemmer()
    review = re.sub("[^a-zA-Z]", " ", text)
    review = review.lower().split()
    review = [stemmer.stem(word) for word in review if word not in STOPWORDS]
    return " ".join(review)

def single_prediction(text_input):
    """Predict sentiment for a single input."""
    processed_text = preprocess_text(text_input)
    X_prediction = cv.transform([processed_text]).toarray()
    X_prediction_scl = scaler.transform(X_prediction)
    y_predictions = predictor.predict_proba(X_prediction_scl)
    return "Positive" if y_predictions.argmax(axis=1)[0] == 1 else "Negative"

def bulk_prediction(data):
    """Predict sentiment for multiple inputs in bulk."""
    corpus = data["Sentence"].apply(preprocess_text)
    X_prediction = cv.transform(corpus).toarray()
    X_prediction_scl = scaler.transform(X_prediction)
    y_predictions = predictor.predict_proba(X_prediction_scl).argmax(axis=1)
    data["Predicted sentiment"] = ["Positive" if x == 1 else "Negative" for x in y_predictions]
    
    # Save predictions to a CSV
    predictions_csv = BytesIO()
    data.to_csv(predictions_csv, index=False)
    predictions_csv.seek(0)
    
    graph = get_distribution_graph(data)
    return predictions_csv, graph

def get_distribution_graph(data):
    """Generate a pie chart of sentiment distribution."""
    fig = plt.figure(figsize=(5, 5))
    colors = ("green", "red")
    wp = {"linewidth": 1, "edgecolor": "black"}
    tags = data["Predicted sentiment"].value_counts()
    explode = (0.01, 0.01)

    tags.plot(
        kind="pie", 
        autopct="%1.1f%%", 
        shadow=True, 
        colors=colors, 
        startangle=90, 
        wedgeprops=wp, 
        explode=explode, 
        title="Sentiment Distribution", 
        xlabel="", 
        ylabel=""
    )

    graph = BytesIO()
    plt.savefig(graph, format="png")
    plt.close(fig)  # Ensure the figure is closed properly
    return graph

if __name__ == "__main__":
    app.run(port=5000, debug=True)
