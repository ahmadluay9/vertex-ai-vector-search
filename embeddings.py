# Import Libraries
import google.auth
from google.cloud import storage, aiplatform
import os
import tensorflow as tf
import tensorflow_hub as hub
import json
import datetime
import dotenv
import logging

from vector_search_image_query import create_gcs_bucket, \
    extract_images_from_pdfs_in_gcs, \
    list_gcs_image_paths, \
    generate_embedding

dotenv.load_dotenv()

# Set the GOOGLE_APPLICATION_CREDENTIALS environment variable
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv('SERVICE_ACCOUNT_PATH')

# Configuration

credentials, project_id = google.auth.default()

PROJECT_ID = project_id
REGION = "asia-southeast2"

# Directory structure
BATCH_ROOT = "batch_root"
EMBEDDINGS_FILE_JSON = 'embeddings.json'
DELETE_DIRECTORY = os.path.join(BATCH_ROOT, "delete")
PDF_FOLDER = "pdf_files"

# GCS Bucket Name and path for embeddings
EMBEDDING_BUCKET_NAME = "sbi-ai-solution-image-embeddings" 
PDF_BUCKET_NAME = f"sample-pdf-{PROJECT_ID}"
IMAGE_BUCKET_NAME = f"{PROJECT_ID}-extracted-images-from-pdf"
GCS_BATCH_ROOT = "batch_root"

INDEX_NAME = "sbi-image-index-01"
IMAGE_DIRECTORY = "images"
MODEL_URL = "https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/5"
NUM_NEIGHBORS = 3

# Initialize Vertex AI SDK
aiplatform.init(project=PROJECT_ID, location=REGION)

# Initialize Google Cloud Storage client
storage_client = storage.Client(project=PROJECT_ID)

# Added Logging Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

"""# 1. Prepare Image Dataset"""

# Ensure required directories exist
os.makedirs(BATCH_ROOT, exist_ok=True)
os.makedirs(DELETE_DIRECTORY, exist_ok=True)
logger.info(f"Created directories: {BATCH_ROOT}, {DELETE_DIRECTORY}")

# Create or get the GCS bucket
embedding_bucket = create_gcs_bucket(EMBEDDING_BUCKET_NAME, REGION, storage_client)
pdf_bucket = create_gcs_bucket(PDF_BUCKET_NAME, REGION, storage_client)
image_bucket = create_gcs_bucket(IMAGE_BUCKET_NAME, REGION, storage_client)
logger.info(f"GCS buckets initialized: embedding: {embedding_bucket.name}, pdf: {pdf_bucket.name}, image: {image_bucket.name}")

# Extract image from pdf
extract_images_from_pdfs_in_gcs(PDF_BUCKET_NAME,IMAGE_BUCKET_NAME)

# Get image paths from the GCS bucket
image_paths = list_gcs_image_paths(image_bucket, IMAGE_DIRECTORY)
if not image_paths:
    logger.error(f"No images found in gs://{image_bucket.name}/{IMAGE_DIRECTORY}. Please put some images in this GCS folder to continue.")
    exit()  # Stop further execution. You have to place images into the folder.

logger.info(f"Found {len(image_paths)} images in the GCS bucket: gs://{image_bucket.name}/{IMAGE_DIRECTORY}")

"""# 2. Generate Image Embeddings"""
# Load a pre-trained ResNet model from TensorFlow Hub
model = hub.KerasLayer(MODEL_URL)
logger.info(f"Loaded pre-trained model from: {MODEL_URL}")

# Generate Embeddings
embeddings = []

for i, path in enumerate(image_paths):
    embedding = generate_embedding(path, image_bucket, model)
    # Extract file name with extension from GCS path
    file_name_with_extension = os.path.basename(path)

    embedding_record = {
        "id": file_name_with_extension,
        "embedding": embedding.tolist()
    }
    embeddings.append(embedding_record)
    logger.info(f"Generated embedding for image: {os.path.basename(path)}")

# Save embeddings to JSON file, writing each record on its own line
embeddings_file_path = os.path.join(BATCH_ROOT, EMBEDDINGS_FILE_JSON)
with open(embeddings_file_path, "w") as f:
    for record in embeddings:
        json.dump(record, f)
        f.write("\n")
logger.info(f"Generated and saved embeddings to: {embeddings_file_path}")

# Upload batch_root to GCS
for root, _, files in os.walk(BATCH_ROOT):
    for file in files:
        local_path = os.path.join(root, file)
        relative_path = os.path.relpath(local_path, BATCH_ROOT)
        blob_path = os.path.join(GCS_BATCH_ROOT, relative_path)
        blob = embedding_bucket.blob(blob_path)
        blob.upload_from_filename(local_path)
        logger.info(f"Uploaded {local_path} to gs://{embedding_bucket.name}/{blob_path}")

# Construct the GCS URI for batch_root
GCS_BATCH_ROOT_URI = f"gs://{EMBEDDING_BUCKET_NAME}/{GCS_BATCH_ROOT}"
logger.info(f"Uploaded batch root to: {GCS_BATCH_ROOT_URI}")

"""# 3. Create Vertex Matching Engine Index"""
# Create a vector index for embeddings
index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
    display_name=INDEX_NAME,
    contents_delta_uri=GCS_BATCH_ROOT_URI,
    dimensions=len(embeddings[0]["embedding"]),
    approximate_neighbors_count=150,
    shard_size="SHARD_SIZE_SMALL"
)

logger.info(f"Created index: {index.display_name}")

# Create the index endpoint
INDEX_ENDPOINT_NAME = f'{INDEX_NAME}-endpoint'

index_endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
  display_name=INDEX_ENDPOINT_NAME,
  public_endpoint_enabled=True
)

logger.info(f"Created index endpoint: {index_endpoint.display_name}")

#  Deploy the created index.
timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
deployed_index_id = f"indexid_sbi_01_{timestamp}"

index_endpoint = index_endpoint.deploy_index(
  index=index,
  deployed_index_id=deployed_index_id,
  machine_type="e2-standard-2",
  min_replica_count=1,
  max_replica_count=1
)

logger.info(f'Deployed endpoint: {index_endpoint.display_name} with ID: {deployed_index_id}')

