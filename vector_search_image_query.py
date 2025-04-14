# Import Libraries
from google.cloud import storage
from google.cloud.aiplatform.gapic import IndexEndpointServiceClient
import os
import numpy as np
from PIL import Image
import fitz  # PyMuPDF
import io
from IPython.display import display, Image as IPImage
import dotenv
import streamlit as st
import logging

dotenv.load_dotenv()

# Set the GOOGLE_APPLICATION_CREDENTIALS environment variable
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv('SERVICE_ACCOUNT_PATH')

"""# 1. Prepare Image Dataset"""

def create_gcs_bucket(bucket_name, region, storage_client):
    """Creates a new GCS bucket if it does not exist.

    Args:
        bucket_name (str): The name of the GCS bucket.
        region (str): The region of the GCS bucket.
        storage_client: A google cloud storage client
    Returns:
        google.cloud.storage.bucket.Bucket: The created or existing bucket.
    """
    try:
        bucket = storage_client.bucket(bucket_name)
        bucket.reload()  # Check if the bucket exists, raises NotFound if not
        print(f"Bucket {bucket.name} already exists.")
        return bucket
    except Exception as e:
        if '404' in str(e): #NotFound exception is not easily accessed
            bucket = storage_client.create_bucket(bucket_name, location=region)
            print(f"Bucket {bucket.name} created.")
            return bucket
        else:
            print(f"Error: {e}")
            raise

def extract_images_from_pdf(pdf_path, image_dir, output_bucket):
    """
    Extracts images from a PDF file and saves them to the specified GCS bucket.

    Args:
        pdf_path (str): The path to the PDF file.
        output_bucket (google.cloud.storage.bucket.Bucket): The GCS bucket where images will be saved.
    """

    try:
        pdf_document = fitz.open(pdf_path)
        image_count = 0

        for page_num in range(pdf_document.page_count):
            page = pdf_document[page_num]
            image_list = page.get_images(full=True)  # Get all images (full path in pdf)

            for image_index, img in enumerate(image_list):
                xref = img[0]  # xref is unique identifier for the image within PDF
                base_image = pdf_document.extract_image(xref)  # extract the actual image
                image_bytes = base_image["image"]

                try:
                    pil_image = Image.open(io.BytesIO(image_bytes))

                    # Generate a unique name based on the PDF filename and image location
                    pdf_filename = os.path.splitext(os.path.basename(pdf_path))[0]
                    image_name = f"{pdf_filename}_page_{page_num + 1}_img_{image_index + 1}.jpg"

                    # Convert to RGB before saving as JPEG in memory
                    pil_image = pil_image.convert('RGB')
                    image_buffer = io.BytesIO()
                    pil_image.save(image_buffer, "JPEG")
                    image_buffer.seek(0)  # Reset buffer to beginning

                    # Upload directly to GCS
                    blob = output_bucket.blob(f"{image_dir}/{image_name}")
                    blob.upload_from_file(image_buffer, content_type="image/jpeg")
                    image_count += 1

                except Exception as e:
                    print(f"Error saving image from page {page_num+1} img {image_index+1} in {pdf_path}: {e}")
                    continue
        print(f"Extracted {image_count} images from {pdf_path} and saved to GCS bucket: gs://{output_bucket.name}/{image_dir}")
    except Exception as e:
        print(f"Error processing PDF {pdf_path}: {e}")


def extract_images_from_pdfs_in_gcs(pdf_bucket_name, output_bucket_name):
    """
    Extracts images from all PDF files in a GCS bucket and saves them to another GCS bucket.
    Saves images to gs://output_bucket_name/images

    Args:
        pdf_bucket_name (str): The name of the GCS bucket containing the PDF files.
        output_bucket_name (str): The name of the GCS bucket where images will be saved.
    """

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    try:
        storage_client = storage.Client()
        pdf_bucket = storage_client.bucket(pdf_bucket_name)
        output_bucket = storage_client.bucket(output_bucket_name)

        logging.info(f"Starting image extraction from PDFs in gs://{pdf_bucket_name} to gs://{output_bucket_name}/images")

        pdf_blobs = pdf_bucket.list_blobs()

        for pdf_blob in pdf_blobs:
            if not pdf_blob.name.lower().endswith(".pdf"):
                logging.info(f"Skipping non-PDF file: {pdf_blob.name}")
                continue

            logging.info(f"Processing PDF: {pdf_blob.name}")

            try:
                # Download PDF to memory
                pdf_bytes = pdf_blob.download_as_bytes()
                pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")

                image_count = 0

                for page_num in range(pdf_document.page_count):
                    page = pdf_document[page_num]
                    image_list = page.get_images(full=True)

                    for image_index, img in enumerate(image_list):
                        xref = img[0]
                        base_image = pdf_document.extract_image(xref)
                        image_bytes = base_image["image"]

                        try:
                            pil_image = Image.open(io.BytesIO(image_bytes))

                            # Generate a unique name based on the PDF filename and image location
                            pdf_filename = os.path.splitext(os.path.basename(pdf_blob.name))[0]
                            image_name = f"{pdf_filename}_page_{page_num + 1}_img_{image_index + 1}.jpg"

                            # Convert to RGB before saving as JPEG in memory
                            pil_image = pil_image.convert('RGB')
                            image_buffer = io.BytesIO()
                            pil_image.save(image_buffer, "JPEG")
                            image_buffer.seek(0)
                            
                            # Upload directly to GCS
                            blob = output_bucket.blob(f"images/{image_name}")
                            blob.upload_from_file(image_buffer, content_type="image/jpeg")
                            image_count += 1
                            logging.info(f"   Saved image to gs://{output_bucket_name}/images/{image_name}")

                        except Exception as e:
                            logging.error(
                                f"   Error saving image from page {page_num + 1} img {image_index + 1} of {pdf_blob.name}: {e}"
                            )
                            continue

                logging.info(f"   Extracted {image_count} images from {pdf_blob.name} and saved to GCS bucket: gs://{output_bucket_name}/images/")

            except Exception as e:
                logging.error(f"   Error processing PDF {pdf_blob.name}: {e}")

        logging.info(f"Finished processing all PDFs in gs://{pdf_bucket_name}")
    except Exception as e:
        logging.error(f"Error accessing GCS buckets: {e}")


def list_gcs_image_paths(bucket, image_dir):
    """Lists all image paths from a GCS bucket folder."""
    image_paths = []
    blobs = bucket.list_blobs(prefix=image_dir)
    for blob in blobs:
        if blob.name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            image_paths.append(f"gs://{bucket.name}/{blob.name}")
    return image_paths


"""# 2. Generate Image Embeddings"""

def generate_embedding(image_path, bucket, model):
    """Generates the embedding for a given image path, whether local or GCS."""
    if image_path.startswith("gs://"):
        # Download from GCS if it's a GCS path
         blob_name = image_path.replace(f"gs://{bucket.name}/", "")
         blob = bucket.blob(blob_name)
         image_bytes = blob.download_as_bytes()
         img = Image.open(io.BytesIO(image_bytes)).resize((224, 224))
    else: # If its a local path
      img = Image.open(image_path).resize((224, 224))
    img = np.array(img) / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    embedding = model(img)
    return embedding.numpy().flatten()

"""# 3. Query the Index"""

def query_index(query_image_path, bucket, model, index_endpoint, deployed_index_id, num_neighbors=3):
  """Queries the Vertex Matching Engine index with a given image."""

  query_embedding = generate_embedding(query_image_path, bucket, model)
  response = index_endpoint.find_neighbors(
      deployed_index_id = deployed_index_id,
      queries=[query_embedding],
      num_neighbors=num_neighbors
  )
  return response

def display_results(response, bucket, image_dir):
    """Displays the search results including the images."""
    print(f"Similar images found:")
    if response and response[0]:  # Check if response is not empty and contains results
        for neighbor in response[0]:
           print(f"  - ID: {neighbor.id}, Distance: {neighbor.distance}")

           # Download the image from GCS to display
           blob_name = f"{image_dir}/{neighbor.id}"
           blob = bucket.blob(blob_name)
           image_bytes = blob.download_as_bytes()
           display(IPImage(data=image_bytes))
    else:
        print("  - No similar images found.")

def display_results_st(response, bucket, image_dir):
    """
    Displays the search results including the images in the Streamlit app in 3 columns.
    """
    st.write("### Similar Images Found:")

    if response and response[0]:  # Check if response is not empty and contains results
        results = response[0]  # Assuming response[0] contains the list of neighbors
        columns = st.columns(3)  # Create 3 columns for layout
        
        for i, neighbor in enumerate(results):
            col = columns[i % 3]  # Cycle through columns
            with col:
                st.write(f"**ID**: {neighbor.id}, **Distance**: {neighbor.distance}")
                
                # Download the image from GCS to display
                blob_name = f"{image_dir}/{neighbor.id}"
                blob = bucket.blob(blob_name)
                image_bytes = blob.download_as_bytes()
                
                # Display the image using Streamlit
                image = Image.open(io.BytesIO(image_bytes)).resize((450, 450))
                st.image(image, caption=f"Distance: {neighbor.distance}", use_container_width=True)
    else:
        st.write("No similar images found.")

def list_index_endpoints_(project_id: str, location: str):
    """Lists all deployed index endpoints in the given project and location."""
    index_endpoint_client = IndexEndpointServiceClient(client_options={"api_endpoint": f"{location}-aiplatform.googleapis.com"})
    parent = f"projects/{project_id}/locations/{location}"
    index_endpoints = index_endpoint_client.list_index_endpoints(parent=parent)

    endpoints_info = []

    for index_endpoint in index_endpoints:
        endpoint_info = {
            "index_endpoint_name": index_endpoint.display_name,
            "deployed_index_id": [
                deployed_index.id for deployed_index in index_endpoint.deployed_indexes
            ],
            "index_endpoint_resource_name": index_endpoint.name,
        }
        endpoints_info.append(endpoint_info)

    return endpoints_info