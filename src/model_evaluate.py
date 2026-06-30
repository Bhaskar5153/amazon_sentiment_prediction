# confusion matrix, precision, recall, f1-score. ----> classification model (binary or multi classification)

import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score, accuracy_score
from pathlib import Path
import joblib

def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        'y_pred': y_pred,
        'accuracy': round(accuracy_score(y_test, y_pred), 4),
        'f_macro': round(f1_score(y_test, y_pred, average="macro"), 4),
        'precision': round(precision_score(y_test, y_pred, average="macro"), 4),
        'recall': round(recall_score(y_test, y_pred, average="macro"), 4)
    }


def classification_report_df(y_test, y_pred, labels=None):
    report = classification_report(y_test, y_pred, target_names=labels, output_dict=True)
    return pd.DataFrame(report).T.round(3)


def confusion_matrix_df(y_test, y_pred, classes):
    cm = confusion_matrix(y_test, y_pred)
    return pd.DataFrame(
        cm,
        index=[f'Actual: {c}' for c in classes],
        columns=[f'Predicted: {c}' for c in classes]
    )




X_test = joblib.load("notebooks/artifacts/X_test.pkl")
y_test = joblib.load("notebooks/artifacts/y_test.pkl")  


def run_all_methods():
    model_path = Path("models")
    models = model_path.glob("*.pkl")
    for model in models:
        if model.name not in ["tfidf_vectorizer.pkl", "label_encoder.pkl"]:
            print(f"Predicted by {model.name}")
            m = joblib.load(model)
            eval = evaluate(model=m, X_test=X_test, y_test=y_test)
            # print(eval)
            predictions = eval['y_pred']


    
    cr = classification_report_df(y_test=y_test, y_pred=predictions)
    # print(cr)

    confusion_m = confusion_matrix_df(y_test=y_test, y_pred=predictions, classes=[0, 1, 2])
    print(confusion_m)


        
    
    
if __name__ == "__main__":
    run_all_methods()



