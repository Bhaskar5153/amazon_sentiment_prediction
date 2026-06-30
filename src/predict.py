import os
import pandas as pd
import numpy as np
import joblib

import re
from pathlib import Path
from scipy.sparse import hstack, csr_matrix

# import ipdb

class Predictor:
    def __init__(self, model_dir: str):
        model_dir = Path(model_dir)
        # ipdb.set_trace()
        self.models = {}
        self.vectorizer = joblib.load(model_dir / "tfidf_vectorizer.pkl") # models\tfidf_vectorizer.pkl
        self.label_encoder = joblib.load(model_dir / "label_encoder.pkl")
        # load the models
        for file in sorted(model_dir.glob("*.pkl")):
            if file.name in ["tfidf_vectorizer.pkl", "label_encoder.pkl"]:
                continue
            model_name = file.name
            self.models[model_name] = joblib.load(file)
            print(f"Loaded models: {list(self.models.keys())}")

        
    # def transform_input(self, raw_texts, numeric_features):
    #     X_text = self.vectorizer.transform(raw_texts)
    #     X_num = csr_matrix(numeric_features)

    #     X = hstack([X_text, X_num])
    #     return X
    def transform_input(self, raw_text):
        return self.vectorizer.transform(raw_text)

    

    
    def predict(self, review_text, review_score=None):
        X = self.transform_input([review_text])
        predictions = {}
        for model_name, model in self.models.items():
            pred = model.predict(X)[0]
            sentiment_class = self.label_encoder.inverse_transform([pred])[0]
            predictions[model_name] = sentiment_class
            print(f"model name: {model_name}")
        return predictions

    
        


# object of the class for predictions.
# predictor = Predictor(model_dir="models")
# ipdb.set_trace()

# X_test = joblib.load("notebooks/artifacts/X_test.pkl")
# result = predictor.predict(X_test)
# # ipdb.set_trace()
# print(result)



