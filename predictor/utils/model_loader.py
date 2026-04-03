import os
import json
import pickle
import numpy as np
import pandas as pd
import torch
from django.conf import settings
# Fix scipy compatibility for pycox/torchtuples
import scipy.integrate
if not hasattr(scipy.integrate, 'simps'):
    scipy.integrate.simps = getattr(scipy.integrate, 'simpson', None)

from .models import ModelConfig
# Adjust import if needed based on actual structure
from .data import DataConfig 

class ModelService:
    _instance = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = ModelService()
        return cls._instance

    def __init__(self):
        self.model = None
        self.encoders = None
        self.x_mapper = None
        self.config = None
        self.feature_cols = None
        self.numeric_cols = None
        self.categorical_cols = None
        self.model_type = None
        self.prediction_target = None
        
        self.load_artifacts()

    def load_artifacts(self):
        # Path to model files
        model_dir = os.path.join(settings.BASE_DIR, 'predictor', 'model_files')
        
        # Load config
        config_path = os.path.join(model_dir, 'config.json')
        if not os.path.exists(config_path):
            print(f"Config file not found at {config_path}")
            return

        with open(config_path, 'r', encoding='utf-8') as f:
            full_config = json.load(f)
            
        self.config = full_config
        data_config = full_config['data_config']
        model_config_dict = full_config['model_config']
        
        self.feature_cols = data_config['feature_cols']
        self.numeric_cols = data_config['numeric_cols']
        self.categorical_cols = data_config['categorical_cols']
        self.prediction_target = data_config['prediction_target']
        self.model_type = model_config_dict['model_type']
        
        # Load encoders
        with open(os.path.join(model_dir, 'encoders.pkl'), 'rb') as f:
            self.encoders = pickle.load(f)
            
        # Load x_mapper
        with open(os.path.join(model_dir, 'x_mapper.pkl'), 'rb') as f:
            self.x_mapper = pickle.load(f)
            
        # Reconstruct model
        # We need to recreate ModelConfig object
        model_conf = ModelConfig(
            model_type=model_config_dict['model_type'],
            **{k:v for k,v in model_config_dict.items() if k != 'model_type'}
        )
        
        # Determine input features
        # Assuming one-hot or embedding is not used in the model definition itself, 
        # but the input dimension to the model depends on the output of x_mapper.
        # We can try to infer it or use a dummy run if possible, but better to know.
        # In train.py: in_features = dataset.x_train.shape[1]
        # x_mapper output shape determines it.
        # Since we have x_mapper, we can't easily ask it for output dim without data.
        # But we can look at the transformed data structure.
        # Numeric cols: 1 each. Categorical: 1 each (LabelEncoded) if passthrough.
        # In dataset.py: categorical cols are 'passthrough', so they stay as 1 col each.
        # So in_features = len(numeric_cols) + len(categorical_cols).
        in_features = len(self.numeric_cols) + len(self.categorical_cols)
        
        self.model, self.net, _ = model_conf.create_model(
            in_features=in_features,
            prediction_target=self.prediction_target,
            device='cpu' # Use CPU for inference usually
        )
        
        # Load weights
        # Find the model file. train.py uses get_model_filename
        model_filename = 'coxph_model.pkl' # Default for survival
        if self.prediction_target == 'event_only':
            model_filename = 'classification_model.pkl'
        elif self.prediction_target == 'time_only':
            model_filename = 'regression_model.pkl'
            
        model_path = os.path.join(model_dir, model_filename)
        if os.path.exists(model_path):
            self.model.load_model_weights(model_path)
        else:
            print(f"Model weights not found at {model_path}")

    def preprocess(self, input_data):
        """
        input_data: dict of {feature_name: value}
        """
        # Create DataFrame
        df = pd.DataFrame([input_data])
        
        # Encode categorical features
        for col, encoder in self.encoders.items():
            if col in df.columns:
                # Handle unseen labels: might raise error or need fallback
                try:
                    df[col] = encoder.transform(df[col].astype(str))
                except ValueError:
                    # Fallback for unseen labels? Or just let it fail?
                    # For now let's try to handle specific errors if needed, 
                    # but encoder.transform usually fails on new labels.
                    # A robust way is to use a special token or mode value.
                    print(f"Warning: Unseen label in column {col}")
                    # Simple fallback: map to 0 or mode (if we knew it)
                    df[col] = 0 
        
        # Ensure all feature columns exist
        for col in self.feature_cols:
            if col not in df.columns:
                df[col] = 0 # or some default
                
        # Select columns in correct order for x_mapper? 
        # ColumnTransformer uses column names, so order in DF shouldn't matter 
        # if columns are passed by name. But let's filter to feature_cols.
        df_subset = df[self.feature_cols]
        
        # Transform
        x = self.x_mapper.transform(df_subset).astype('float32')
        return x

    def predict(self, input_data):
        if self.model is None:
            return {"error": "Model not loaded"}
            
        try:
            x = self.preprocess(input_data)
            
            if self.prediction_target == 'survival':
                 # Compute baseline hazards if not already (might be saved in model weights? 
                 # No, PyCox/torchtuples usually needs training data for baseline hazards 
                 # if not explicitly saved. train.py calls compute_baseline_hazards(x_train, y_train).
                 # This is a PROBLEM if we don't have training data during inference.
                 # Check if the saved model weights include baseline hazards.
                 # CoxPH model in pycox stores baseline hazards in self.duration_index and self.cum_baseline_hazard
                 # IF they were computed. But `load_model_weights` only loads pytorch state dict usually.
                 # We might need to save/load baseline hazards separately or check if torchtuples saves them.
                 pass
            
            # For now, let's try to predict. If it's survival, we want survival probabilities.
            # If baseline hazards are missing, predict_surv_df will fail.
            # Alternatives: Predict risk score (hazard ratio) using model.predict(x)
            
            if self.prediction_target == 'survival':
                # Return risk score (log-hazard)
                risk = self.model.predict(x)
                # Convert to Python list/float
                return {"risk_score": float(risk[0][0])}
                
                # If we want survival curves, we definitely need baseline hazards.
                # The train.py script calculates them using training data.
                # Since we don't have training data here, we can't easily recompute them.
                # We should have saved the baseline hazards or the survival function object.
                # But looking at train.py, it only saves model weights and survival PREDICTIONS for test set.
                # It doesn't seem to serialize the baseline hazards specifically.
                
            elif self.prediction_target == 'event_only':
                logits = self.model.predict(x)
                prob = torch.sigmoid(torch.tensor(logits)).item()
                return {"probability": prob, "prediction": 1 if prob > 0.5 else 0}
                
            elif self.prediction_target == 'time_only':
                pred = self.model.predict(x)
                return {"predicted_time": float(pred[0][0])}
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"error": str(e)}


