from flask import Flask, request, jsonify
import pandas as pd
import joblib
import os
import logging
import logging.handlers
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import logstash
import signal
import sys
from threading import Event

app = Flask(__name__)

# Logging Setup
LOG_FILE = "access.log"
LOGSTASH_HOST = "172.17.0.1"
LOGSTASH_PORT = 5000  # logstash TCP port

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

file_handler = logging.FileHandler(LOG_FILE)
file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(file_formatter)
root_logger.addHandler(file_handler)

try:
    logstash_handler = logstash.TCPLogstashHandler(LOGSTASH_HOST, LOGSTASH_PORT, version=1)
    root_logger.addHandler(logstash_handler)
    root_logger.info("Connected to Logstash successfully.")
except Exception as e:
    root_logger.warning(f"Failed to connect to Logstash: {e}")

app.logger.setLevel(logging.INFO)
app.logger.addHandler(logstash_handler)

shutdown_event = Event()

def graceful_shutdown(signum, frame):
    app.logger.info("Received termination signal, shutting down gracefully...")
    shutdown_event.set()
    sys.exit(0)

signal.signal(signal.SIGTERM, graceful_shutdown)
signal.signal(signal.SIGINT, graceful_shutdown)

# Load all saved models once at startup
MODEL_DIR = "models"
models = {}

for model_file in os.listdir(MODEL_DIR):
    if model_file.endswith(".pkl"):
        model_name = model_file.replace(".pkl", "")
        try:
            models[model_name] = joblib.load(os.path.join(MODEL_DIR, model_file))
            app.logger.info(f"Loaded model '{model_name}' from disk.")
        except Exception as e:
            app.logger.error(f"Failed to load model '{model_name}': {e}")

# Flask Routes

@app.route("/")
def index():
    client_ip = request.remote_addr
    logging.info(f"{client_ip} accessed the root endpoint '/'")
    return "ML Prediction API is running."

@app.route("/healthz")
def health_check():
    return "OK", 200

@app.route("/predict", methods=["POST"])
def predict():
    client_ip = request.remote_addr
    logging.info(f"{client_ip} accessed '/predict' endpoint")

    data = request.json
    model_type = data.get("model", "log_reg")
    input_data = data.get("input")

    if not input_data:
        logging.warning(f"{client_ip} - Missing input data.")
        return jsonify({"error": "Missing input data"}), 400

    if model_type not in models:
        logging.error(f"{client_ip} - Model '{model_type}' not found in loaded models.")
        return jsonify({"error": "Model type not found"}), 400

    model, columns = models[model_type]

    input_df = pd.DataFrame([input_data])
    input_df = pd.get_dummies(input_df)
    input_df = input_df.reindex(columns=columns, fill_value=0)

    prediction = model.predict(input_df)[0]
    logging.info(f"{client_ip} - Prediction made using '{model_type}': {prediction}")
    return jsonify({"prediction": int(prediction)})


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5001)
