from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure

# 1. Define the connection details
# These match the environment variables in your docker-compose.yml
USER = "admin"
PASSWORD = "password123"
HOST = "localhost"
PORT = 27017

# Construct the connection URL
# Format: mongodb://username:password@host:port/
connection_string = f"mongodb://{USER}:{PASSWORD}@{HOST}:{PORT}/"

print(f"🔌 Attempting to connect to MongoDB at {HOST}:{PORT}...")

try:
    # 2. Create the client
    # serverSelectionTimeoutMS limits how long we wait before failing (2 seconds here)
    client = MongoClient(connection_string, serverSelectionTimeoutMS=2000)

    # 3. The "Ping"
    # Simply creating the client doesn't actually connect. 
    # We must run a command to force a connection attempt.
    server_info = client.server_info()
    
    print("\n✅ SUCCESS: Connected to MongoDB!")
    print(f"   Server Version: {server_info['version']}")
    
    # Optional: List databases to prove permissions work
    dbs = client.list_database_names()
    print(f"   Existing Databases: {dbs}")

except ConnectionFailure:
    print("\n❌ ERROR: Could not connect to server.")
    print("   Make sure 'docker-compose up' is running and port 27017 is not blocked.")

except OperationFailure as e:
    print("\n❌ ERROR: Authentication failed.")
    print("   Check if the username/password in this script matches your docker-compose.yml.")
    print(f"   Details: {e}")

except Exception as e:
    print(f"\n❌ ERROR: An unexpected error occurred: {e}")