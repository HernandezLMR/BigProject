import os
import torch
from minio import Minio
from models.inf_model import InferenceModel

MODEL_INSTANCE = None
MINIO_CLIENT = None

def initialize_worker():
    global MODEL_INSTANCE, MINIO_CLIENT
    MODEL_INSTANCE = InferenceModel("/app/models/best_nih_densenet121.pth") 
    MINIO_CLIENT = Minio(
        "minio:9000",
        access_key="minioadmin",
        secret_key="minioadmin",
        secure=False
    )

def process_image_task(doc):
    global MODEL_INSTANCE, MINIO_CLIENT
    
    if MODEL_INSTANCE is None:
        initialize_worker()
        
    filename = doc['filename']
    bucket = doc['minio_bucket']
    object_name = doc['minio_object']
    
    temp_path = f"/tmp/{filename}"
    
    try:
        MINIO_CLIENT.fget_object(bucket, object_name, temp_path)
        
        full_results, high_confidence = MODEL_INSTANCE.predict(temp_path)
        
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        return {
            "status": "success",
            "filename": filename,
            "inference": full_results # Storing just the important classes
        }
        
    except Exception as e:
        # Cleanup even if it failed
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return {"status": "error", "filename": filename, "error": str(e)}