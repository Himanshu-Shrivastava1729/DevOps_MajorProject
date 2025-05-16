from flask import Flask, request, jsonify
import pandas as pd
import joblib
import os
import logging
import logging.handlers
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import logstash
app = Flask(__name__)

# ============================
# Logging Setup
# ============================

LOG_FILE = "access.log"
# LOGSTASH_HOST = "logstash"  # logstash service name from docker-compose
LOGSTASH_HOST = "172.17.0.1"
LOGSTASH_PORT = 5000  # logstash TCP port

# Configure the root logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

# File handler (optional)
file_handler = logging.FileHandler(LOG_FILE)
file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(file_formatter)
root_logger.addHandler(file_handler)

# Logstash handler for JSON logs
try:
    logstash_handler = logstash.TCPLogstashHandler(LOGSTASH_HOST, LOGSTASH_PORT, version=1)
    root_logger.addHandler(logstash_handler)
    root_logger.info("Connected to Logstash successfully.")
except Exception as e:
    root_logger.warning(f"Failed to connect to Logstash: {e}")

# Add logstash handler to Flask's logger as well (so Flask internal logs are captured)
app.logger.setLevel(logging.INFO)
app.logger.addHandler(logstash_handler)

# ============================
# Load, Preprocess, Train
# ============================


def load_and_preprocess():
    logging.info("Loading and preprocessing data...")
    df = pd.read_csv("Assignment-2_Data.csv")

    # Convert target to binary
    df["y"] = df["y"].apply(lambda x: 1 if x == "yes" else 0)

    # Remove outliers
    df = df[df["age"] <= 100]
    for col in ["age", "balance"]:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]

    # One-hot encoding
    df = pd.get_dummies(
        df, columns=["job", "marital", "month", "poutcome", "education", "contact"]
    )

    # Encode binary columns
    for col in ["default", "housing", "loan"]:
        df[col] = df[col].map({"yes": 1, "no": 0})

    logging.info("Data preprocessing complete.")
    return df


def train_models(df):
    logging.info("Starting model training...")
    X = df.drop("y", axis=1)
    y = df["y"]

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "log_reg": LogisticRegression(max_iter=1000),
        "gnb": GaussianNB(),
        "dt": DecisionTreeClassifier(),
    }

    os.makedirs("models", exist_ok=True)

    for name, model in models.items():
        model.fit(X_train, y_train)
        joblib.dump((model, X.columns), f"models/{name}.pkl")
        logging.info(f"Trained and saved model: {name}")

    logging.info("All models trained and saved successfully.")


# ============================
# Flask Routes
# ============================


@app.route("/")
def index():
    client_ip = request.remote_addr
    logging.info(f"{client_ip} accessed the root endpoint '/'")
    return "ML Prediction API is running."


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

    try:
        model, columns = joblib.load(f"models/{model_type}.pkl")
    except Exception as e:
        logging.error(
            f"{client_ip} - Model '{model_type}' not found or failed to load. Error: {str(e)}"
        )
        return jsonify({"error": "Model type not found"}), 400

    input_df = pd.DataFrame([input_data])
    input_df = pd.get_dummies(input_df)
    input_df = input_df.reindex(columns=columns, fill_value=0)

    prediction = model.predict(input_df)[0]
    logging.info(f"{client_ip} - Prediction made using '{model_type}': {prediction}")
    return jsonify({"prediction": int(prediction)})


# ============================
# Train models on startup
# ============================

if __name__ == "__main__":
    df = load_and_preprocess()
    train_models(df)
    app.run(host="0.0.0.0", port=5001)
