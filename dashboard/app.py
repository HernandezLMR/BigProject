import streamlit as st
import pandas as pd
from minio import Minio
from pymongo import MongoClient
import io
from datetime import datetime
import time
import altair as alt

MONGO_URI = "mongodb://admin:password123@localhost:27017/"
MINIO_CONF = {
    "endpoint": "localhost:9000",
    "access_key": "minioadmin",
    "secret_key": "minioadmin",
    "secure": False
}
BUCKET_NAME = "raw-images"


@st.cache_resource
def get_mongo_collection():
    client = MongoClient(MONGO_URI)
    return client["ml_metadata"]["image_queue"]

@st.cache_resource
def get_minio_client():
    return Minio(**MINIO_CONF)


def upload_files(uploaded_files):
    minio_client = get_minio_client()
    mongo_collection = get_mongo_collection()
    
    if not minio_client.bucket_exists(BUCKET_NAME):
        minio_client.make_bucket(BUCKET_NAME)

    progress_bar = st.progress(0)
    
    for i, file in enumerate(uploaded_files):
        bytes_data = file.getvalue()
        file_stream = io.BytesIO(bytes_data)
        
        object_name = file.name 
        
        minio_client.put_object(
            BUCKET_NAME,
            object_name,
            file_stream,
            length=len(bytes_data),
            content_type=file.type
        )
        
        doc = {
            "filename": file.name,
            "minio_bucket": BUCKET_NAME,
            "minio_object": object_name,
            "status": "pending_processing",
            "uploaded_at": datetime.now(),
            "inference_result": {} # Initialize as empty
        }
        
        mongo_collection.update_one(
            {"filename": file.name}, 
            {"$set": doc}, 
            upsert=True
        )
        
        progress_bar.progress((i + 1) / len(uploaded_files))
        
    st.success(f"Uploaded {len(uploaded_files)} images to queue")
    time.sleep(1)
    st.rerun()

def get_image_bytes(bucket, object_name):
    client = get_minio_client()
    try:
        response = client.get_object(bucket, object_name)
        return response.read()
    except Exception as e:
        st.error(f"Error loading image: {e}")
        return None

#UI Setup
st.set_page_config(page_title="X-Ray Analysis Dashboard", layout="wide")

st.title("X-Ray Analysis Dashboard")


with st.sidebar:
    st.header("Upload New Scans")
    uploaded_files = st.file_uploader(
        "Drop X-Rays here", 
        type=['png', 'jpg', 'jpeg'], 
        accept_multiple_files=True
    )
    
    if uploaded_files:
        if st.button("Process Images"):
            upload_files(uploaded_files)
    


collection = get_mongo_collection()

#fetch data
cursor = collection.find().sort("uploaded_at", -1)
data = list(cursor)

if not data:
    st.write("No data found. Upload images to start.")
else:
    df = pd.DataFrame(data)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Images", len(df))
    col2.metric("Processed", len(df[df['status'] == 'processed']))
    col3.metric("Pending", len(df[df['status'] == 'pending_processing']))

    st.divider()

    st.subheader("Recent Analysis")
    
    display_df = df[['filename', 'status', 'uploaded_at']].copy()
    
    selected_row = st.dataframe(
        display_df, 
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row"
    )

    if selected_row and len(selected_row.selection['rows']) > 0:
        index = selected_row.selection['rows'][0]
        selected_item = df.iloc[index]
        
        st.subheader(f"Detail View: {selected_item['filename']}")
        
        detail_col1, detail_col2 = st.columns([1, 2])
        
        with detail_col1:
            img_bytes = get_image_bytes(selected_item['minio_bucket'], selected_item['minio_object'])
            if img_bytes:
                st.image(img_bytes, caption=selected_item['filename'], use_container_width=True)

        with detail_col2:
            st.write(f"**Status:** `{selected_item['status']}`")
            
            if selected_item['status'] == 'processed':
                st.write("### Model Predictions")
                
                results = selected_item.get('inference_result', {})
                if results:
                    sorted_res = dict(sorted(results.items(), key=lambda item: item[1], reverse=True))
                    
                    chart_data = pd.DataFrame({
                        'Condition': list(sorted_res.keys()),
                        'Probability': list(sorted_res.values())
                    })
                    chart = alt.Chart(chart_data).mark_bar().encode(
                        x=alt.X('Probability', scale=alt.Scale(domain=[0, 1])),
                        y=alt.Y('Condition', sort='-x'),                       
                        tooltip=['Condition', alt.Tooltip('Probability', format='.1%')] 
                    ).properties(
                        title="Probability Distribution of Detected Conditions"
                    )
                    
                    st.altair_chart(chart, use_container_width=True)
                else:
                    st.warning("No conditions detected above threshold.")
            elif selected_item['status'] == 'pending_processing':
                st.info("Waiting for worker...")
            else:
                st.error(f"Processing Failed: {selected_item.get('error', 'Unknown error')}")

    if st.button("Refresh Data"):
        st.rerun()