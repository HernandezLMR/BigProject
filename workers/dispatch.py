from dask.distributed import Client
from pymongo import MongoClient

def main():
    # Connect toDask Cluster
    client = Client("localhost:8786")
    print(f"Connected to Dask: {client}")

    # Connect to Mongo
    mongo = MongoClient("mongodb://admin:password123@localhost:27017/")
    db = mongo["ml_metadata"]
    collection = db["image_queue"]

    # Fetch Pending Documents
    pending_docs = list(collection.find({"status": "pending_processing"}).limit(100))
    
    if not pending_docs:
        print("No pending documents found.")
        return

    print(f"Submitting {len(pending_docs)} tasks...")

    # Submit tasks
    from worker_logic import process_image_task

    futures = []
    for doc in pending_docs:
        # map/submit sends the function and data to the worker
        future = client.submit(process_image_task, doc)
        futures.append(future)

    results = client.gather(futures)

    # Update database
    for res in results:
        if res['status'] == 'success':
            collection.update_one(
                {"filename": res['filename']},
                {"$set": {"status": "processed", "result": res['inference']}}
            )
            print(f" Processed: {res['filename']}")
        else:
            print(f" Failed: {res['filename']} - {res['error']}")

if __name__ == "__main__":
    main()