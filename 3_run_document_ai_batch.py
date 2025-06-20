from google.cloud import documentai_v1 as documentai
from google.api_core.client_options import ClientOptions

# Configuration (change accordingly)
project_id = <your_project_id>
location = <location_of_the_project>
processor_id = <your_processor_id>
gcs_input_uri = ".../minneapolis_1910_input/"
gcs_output_uri = ".../minneapolis_1910_output/"

# Initialize Document AI Client
opts = ClientOptions(api_endpoint=f"us-documentai.googleapis.com")
docai_client = documentai.DocumentProcessorServiceClient(client_options=opts)

# Processor resource name
processor_name = docai_client.processor_path(project_id, location, processor_id)

# Configure Batch Process Request correctly without mime_type
batch_request = documentai.BatchProcessRequest(
    name=processor_name,
    input_documents=documentai.BatchDocumentsInputConfig(
        gcs_prefix=documentai.GcsPrefix(gcs_uri_prefix=gcs_input_uri)
    ),
    document_output_config=documentai.DocumentOutputConfig(
        gcs_output_config=documentai.DocumentOutputConfig.GcsOutputConfig(
            gcs_uri=gcs_output_uri
        )
    )
)

# Submit the request
operation = docai_client.batch_process_documents(request=batch_request)

print("Document AI OCR operation started. Processing asynchronously...")
print(f"Operation ID: {operation.operation.name}")

# Wait for the operation to finish
operation.result(timeout=1800)  # 30 minutes timeout (adjust as needed)

print("Document AI OCR completed. Check output bucket for results.")
