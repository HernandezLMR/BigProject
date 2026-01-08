import time
from dask.distributed import Client, as_completed
from pymongo import MongoClient

def remote_runner(doc):
    from worker_logic import process_image_task
    
    return process_image_task(doc)


def main():
    # Establish Connections
    print("Connecting to Dask...")
    client = Client("tcp://localhost:8786")
    
    print("Connecting to MongoDB...")
    mongo = MongoClient("mongodb://admin:password123@localhost:27017/")
    db = mongo["ml_metadata"]
    collection = db["image_queue"]

    # Get Pending Tasks
    batch_size = 10
    cursor = collection.find({"status": "pending_processing"}).limit(batch_size)
    docs = list(cursor)
    
    if not docs:
        print("No pending tasks found")
        return

    print(f"Dispatching {len(docs)} tasks to the cluster...")

    # Submit Tasks
    futures = []
    for doc in docs:
        # Convert the MongoDB ObjectId to a simple string
        if '_id' in doc:
            doc['_id'] = str(doc['_id'])
        
        if 'uploaded_at' in doc:
             doc['uploaded_at'] = doc['uploaded_at'].isoformat()

        future = client.submit(remote_runner, doc)
        futures.append(future)

    print("Waiting for results...")
    
    for future in as_completed(futures):
        result = future.result() 
        
        filename = result.get('filename')
        status = result.get('status')

        if status == "success":
            raw_inference = result['inference']
            
            clean_inference = {k: float(v) for k, v in raw_inference.items()}

            collection.update_one(
                {"filename": filename},
                {
                    "$set": {
                        "status": "processed",
                        "inference_result": clean_inference,
                        "processed_at": time.time()
                    }
                }
            )
            print(f"Finished: {filename}")
            
        else:
            error_msg = result.get('error')
            print(f"Failed: {filename} - {error_msg}")
            collection.update_one(
                {"filename": filename},
                {"$set": {"status": "failed", "error": error_msg}}
            )

    print("Batch complete.")

if __name__ == "__main__":
    while True:
        main()
        time.sleep(5)  # Wait before checking for new tasks