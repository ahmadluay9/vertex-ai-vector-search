# Import Libraries
import streamlit as st
import google.auth
from google.cloud import storage, aiplatform
import os
import tensorflow_hub as hub
import dotenv
from vector_search_image_query import create_gcs_bucket, \
    query_index, \
    display_results_st, \
    list_index_endpoints_

dotenv.load_dotenv()

# Set the GOOGLE_APPLICATION_CREDENTIALS environment variable
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv('SERVICE_ACCOUNT_PATH')

# Configuration
credentials, project_id = google.auth.default()
PROJECT_ID = project_id
REGION = "asia-southeast2" 

# Directory structure and GCS settings
PDF_FOLDER = "pdf_files"
GCS_IMAGE_BUCKET_NAME = f"{PROJECT_ID}-extracted-images-from-pdf"  
GCS_BATCH_ROOT = "batch_root"
IMAGE_DIRECTORY = "images" # Local directory, will be replaced with GCS later
MODEL_URL = "https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/5"
NUM_NEIGHBORS = 3

# Initialize Vertex AI SDK and GCS Client
aiplatform.init(project=PROJECT_ID, location=REGION)
storage_client = storage.Client(project=PROJECT_ID)

# Create or get the GCS bucket
bucket = create_gcs_bucket(GCS_IMAGE_BUCKET_NAME, REGION, storage_client)

# Load a pre-trained ResNet model from TensorFlow Hub
model = hub.KerasLayer(MODEL_URL)

# List Deployed Endpoints
list_endpoint = list_index_endpoints_(PROJECT_ID, REGION)

index_endpoint_name  = list_endpoint[0]["index_endpoint_name"]
deployed_index_id = list_endpoint[0]["deployed_index_id"]
deployed_index_id = deployed_index_id[0]
index_endpoint_resource_name = list_endpoint[0]["index_endpoint_resource_name"]

# Get the index endpoint object from the name
index_endpoint = aiplatform.MatchingEngineIndexEndpoint(index_endpoint_resource_name)

# Streamlit app
uploaded_file = st.file_uploader("Upload an image to find similar images", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Ensure the IMAGE_DIRECTORY exists
    if not os.path.exists(IMAGE_DIRECTORY):
        os.makedirs(IMAGE_DIRECTORY)
    
    query_image_path = os.path.join(IMAGE_DIRECTORY, uploaded_file.name)
    with open(query_image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.image(query_image_path, caption="Uploaded Image", use_container_width=True)
    st.write("Querying the Matching Engine Index for similar images...")
    
    try:
        # Query the index
        query_results = query_index(
            query_image_path,
            bucket,
            model,
            index_endpoint,
            deployed_index_id,
            num_neighbors=NUM_NEIGHBORS
        )
        st.success("Query completed successfully!")
        
        # Display results
        st.write("Top similar images:")
        display_results_st(query_results, bucket, IMAGE_DIRECTORY)
    
    except Exception as e:
        st.error(f"An error occurred during the query: {e}")