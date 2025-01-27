import os
import sys
import pandas as pd
import pickle
from flask import request
from src.constant import *
from src.utils.main_utils import MainUtils
from dataclasses import dataclass
from src.logger import logging # Import the logger
from src.exception import CustomException

@dataclass
class PredictionPipelineConfig:
    prediction_output_dirname: str = "predictions"
    prediction_file_name: str = "prediction_file.csv"
    model_file_path: str = os.path.join(artifact_folder, 'model.pkl')
    preprocessor_path: str = os.path.join(artifact_folder, 'preprocessor.pkl')
    prediction_file_path: str = os.path.join(prediction_output_dirname, prediction_file_name)

class PredictionPipeline:
    def __init__(self, request: request):
        self.request = request
        self.utils = MainUtils()
        self.prediction_pipeline_config = PredictionPipelineConfig()

    def save_input_files(self) -> str:
        try:
            pred_file_input_dir = "prediction_artifacts"
            os.makedirs(pred_file_input_dir, exist_ok=True)

            input_csv_file = self.request.files['file']
            pred_file_path = os.path.join(pred_file_input_dir, input_csv_file.filename)
            input_csv_file.save(pred_file_path)

            return pred_file_path
        except Exception as e:
            logger.error(f"Error in save_input_files: {e}")
            raise CustomException(e, sys) from e

    def get_predicted_dataframe(self, input_csv_path: str):
        try:
            # Load the model
            if not os.path.exists(self.prediction_pipeline_config.model_file_path):
                raise FileNotFoundError(f"Model file not found: {self.prediction_pipeline_config.model_file_path}")

            with open(self.prediction_pipeline_config.model_file_path, 'rb') as model_file:
                model = pickle.load(model_file)

            # Load the preprocessor
            if not os.path.exists(self.prediction_pipeline_config.preprocessor_path):
                raise FileNotFoundError(f"Preprocessor file not found: {self.prediction_pipeline_config.preprocessor_path}")

            with open(self.prediction_pipeline_config.preprocessor_path, 'rb') as preprocessor_file:
                preprocessor = pickle.load(preprocessor_file)

            # Read the input CSV
            input_csv_path = r"C:\Users\SAI\Documents\DA\SensorProject\notebooks\wafer_23012020_041211.csv"
            input_df = pd.read_csv(input_csv_path)

            # Preprocess the input data
            input_data = preprocessor.transform(input_df)

            # Make predictions
            predictions = model.predict(input_data)

            # Save predictions to a CSV file
            prediction_df = pd.DataFrame(predictions, columns=['Prediction'])
            prediction_df.to_csv(self.prediction_pipeline_config.prediction_file_path, index=False)

            logger.info("Predictions completed successfully.")
        except Exception as e:
            logger.error(f"Error in get_predicted_dataframe: {e}")
            raise CustomException(e, sys) from e

    def run_pipeline(self):
        try:
            logger.info("Starting run_pipeline method.")
            input_csv_path = self.save_input_files()
            logger.info(f"Input CSV path: {input_csv_path}")

            self.get_predicted_dataframe(input_csv_path)
            logger.info("Prediction completed successfully.")

            return self.prediction_pipeline_config
        except Exception as e:
            logger.error(f"Error in run_pipeline: {e}")
            raise CustomException(e, sys) from e