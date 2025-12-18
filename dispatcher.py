import time
from dask.distributed import Client, as_completed
from pymongo import MongoClient

# ---------------------------------------------------------
# THE SHIM FUNCTION
# This function is defined here but runs ON THE WORKER.
# ---------------------------------------------------------
def remote_runner(doc):
    """
    This wrapper ensures we import the logic using the WORKER'S file structure,
    not the Client's structure.
    """
    # Import inside the function! 
    # The worker has 'worker_logic.py' in its root /app folder.
    from worker_logic import process_image_task
    
    return process_image_task(doc)

# ---------------------------------------------------------
# MAIN DISPATCHER
# ---------------------------------------------------------
def main():
    # 1. Connect to Resources
    print("🔌 Connecting to Dask...")
    client = Client("tcp://localhost:8786")
    
    print("🔌 Connecting to MongoDB...")
    mongo = MongoClient("mongodb://admin:password123@localhost:27017/")
    db = mongo["ml_metadata"]
    collection = db["image_queue"]

    # 2. Get Pending Tasks
    # Fetch tasks that haven't been processed yet
    batch_size = 10
    cursor = collection.find({"status": "pending_processing"}).limit(batch_size)
    docs = list(cursor)
    
    if not docs:
        print("✅ No pending tasks found. Upload more images with ingest.py!")
        return

    print(f"🚀 Dispatching {len(docs)} tasks to the cluster...")

    # 3. Submit Tasks (Fire!)
    # We map the documents to futures (promises of a result)
    futures = []
    for doc in docs:
        # SANITIZATION STEP:
        # Convert the MongoDB ObjectId to a simple string
        if '_id' in doc:
            doc['_id'] = str(doc['_id'])
        
        # If you have datetime objects, convert them to strings too
        if 'uploaded_at' in doc:
             doc['uploaded_at'] = doc['uploaded_at'].isoformat()

        # NOW submit the "clean" dictionary (which is just JSON-compatible data)
        future = client.submit(remote_runner, doc)
        futures.append(future)

    # 4. Handle Results (Real-time)
    # as_completed yields futures as they finish, so you don't wait for the whole batch
    print("⏳ Waiting for results...")
    
    for future in as_completed(futures):
        result = future.result() 
        
        filename = result.get('filename')
        status = result.get('status')

        if status == "success":
            # --- THE FIX STARTS HERE ---
            # 1. Get the raw inference dictionary
            raw_inference = result['inference']
            
            # 2. Convert values from np.float32 -> python float
            #    We use float() to force the conversion
            clean_inference = {k: float(v) for k, v in raw_inference.items()}
            # --- THE FIX ENDS HERE ---

            # 3. Update Mongo with the CLEAN dictionary
            collection.update_one(
                {"filename": filename},
                {
                    "$set": {
                        "status": "processed",
                        "inference_result": clean_inference, # Use the clean version
                        "processed_at": time.time()
                    }
                }
            )
            print(f"✅ Finished: {filename}")
            
        else:
            # Handle Errors
            error_msg = result.get('error')
            print(f"❌ Failed: {filename} - {error_msg}")
            collection.update_one(
                {"filename": filename},
                {"$set": {"status": "failed", "error": error_msg}}
            )

    print("🏁 Batch complete.")

if __name__ == "__main__":
    while True:
        main()
        time.sleep(5)  # Wait before checking for new tasks