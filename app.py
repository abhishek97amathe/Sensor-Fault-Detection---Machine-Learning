from flask import Flask, render_template, jsonify, request, send_file
from src.exception import CustomException
from src.logger import logging as lg
import os, sys

from src.pipeline.train_pipeline import TrainingPipeline
from src.pipeline.predict_pipeline import PredictionPipeline

app = Flask(__name__)

@app.route("/")
def home():
    lg.info("Home route accessed.")
    return "Welcome to my application"

@app.route("/train")
def train_route():
    try:
        lg.info("Training route accessed.")
        train_pipeline = TrainingPipeline()
        train_pipeline.run_pipeline()
        lg.info("Training pipeline completed.")
        return "Training Completed."
    except Exception as e:
        lg.error(f"Error in training route: {str(e)}")
        raise CustomException(e, sys)

@app.route('/predict', methods=['POST', 'GET'])
def upload():
    try:
        if request.method == 'POST':
            prediction_pipeline = PredictionPipeline(request)
            prediction_file_detail = prediction_pipeline.run_pipeline()
            lg.info("Prediction completed. Downloading prediction file.")
            return send_file(
                prediction_file_detail.prediction_file_path,
                download_name=prediction_file_detail.prediction_file_name,
                as_attachment=True,
            )
        else:
            return render_template('upload_file.html')
    except Exception as e:
        lg.error(f"Error in prediction route: {str(e)}")
        raise CustomException(e, sys)

if __name__ == "__main__":
    lg.info("Starting the Flask application.")
    app.run(host="0.0.0.0", port=5000, debug=True)
