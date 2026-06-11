import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from google.cloud import storage
import tempfile
import pandas as pd


PROJECT_ID = "ai-ml-solutions-492011"
LOCATION = "us-central1"
SERVICE_ACCOUNT_PATH = "C:/Users/Priya Bhaskar/Downloads/ai-ml-solutions-492011-0a0ac6efd9b4.json"

bucket_name = "machine_learning_datasets"
blob_name = "raw_data/cleaned_reviews.csv"

# storage_client = storage.Client.from_service_account_json(json_credentials_path=SERVICE_ACCOUNT_PATH)

processed_folder = "processed_data"


# def load_data_from_gcs(file_name):
#     bucket = storage_client.bucket(bucket_name=bucket_name)
#     blob = bucket.blob(blob_name=f"raw_data/{file_name}")
#     print(blob_name)
#     # ipdb.set_trace()

#     local_file_path = os.path.join(tempfile.gettempdir(), file_name)
#     blob.download_to_filename(local_file_path)
#     print(local_file_path)
#     df = pd.read_csv(local_file_path)

#     return df


def load_data_from_gcs(file_name):
    storage_client = storage.Client.from_service_account_json(json_credentials_path=SERVICE_ACCOUNT_PATH)
    bucket = storage_client.bucket(bucket_name)

    # Try a few filename variants to be more robust (with/without .csv)
    candidates = [file_name]
    base = os.path.splitext(file_name)[0]
    if not file_name.endswith('.csv'):
        candidates.append(f"{file_name}.csv")
    else:
        candidates.append(base)  # try without extension too

    last_exc = None
    for candidate in candidates:
        blob_path = f"raw_data/{candidate}"
        blob = bucket.blob(blob_path)
        local_file_path = os.path.join(tempfile.gettempdir(), os.path.basename(candidate))
        try:
            blob.download_to_filename(local_file_path)
            print(f"Downloaded file to: {local_file_path}")
            df = pd.read_csv(local_file_path)
            return df
        except Exception as exc:
            last_exc = exc
            print(f"Could not download '{blob_path}': {exc}")

    # If we reach here, none of the candidates worked
    raise FileNotFoundError(f"Could not find any of {candidates} in bucket '{bucket_name}'") from last_exc



    

# print(load_data_from_gcs("cleaned_reviews.csv"))