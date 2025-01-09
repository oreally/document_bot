# Document Manager and RAG Chatbot
Streamlit app for RAG chat with documents, using GCS, LlamaParse, Qdrant, and the Groq model.

Original sources: https://github.com/divakarkumarp/RAG-Chat-with-Documents/tree/main, https://huggingface.co/ThisIs-Developer/Llama-2-GGML-Medical-Chatbot

However, the code from there has changed a lot.
 
## Overview:
Software And Tools Requirements

1. [Google Cloud Storage](https://console.cloud.google.com/)
2. [LlamaParse](https://cloud.llamaindex.ai/)
3. [Qdrant](https://cloud.qdrant.io/)
4. [Groq](https://groq.com/)
5. [Langchain🦜](https://www.langchain.com/)
6. [Langchain Smith 🦜](https://smith.langchain.com/o/32390bae-a13d-5a53-b61b-501e3f39e496/projects/p/7e7575b9-5a88-46e5-b7d1-819569ebb004?timeModel=%7B%22duration%22%3A%227d%22%7D&tab=0)

## Prepare
1. Create Google-Cloud Storage bucket and service account with update permissions
2. Create Llama-Cloud API-KEY
3. Create Qdrant cluster and API-KEY
4. Create Groq API-KEY
5. Store GCS connection info, API-KEYs and Qdrant cluster endpoint URL (see below) in .streamlit/secrets.toml (for local run) and streamlit.io secrets (https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management) for deployment.

LLAMA_CLOUD_API_KEY="..."  
QDRANT_API_KEY="..."  
QDRANT_URL="..."  
GROQ_API_KEY="..."  
GCS_BUCKET="..."  
GCS_PROTECTED_FOLDER="source_files" # create these folders in the GCS bucket.  
GCS_DESTINATION_FOLDER="documents"  
PARSED_DATA_FILE="parsed_data.pkl" # do not create, is done by parsing the first time.  
PARSED_OUTPUT_FILE="output.md"  

[connection.gcs] # get these values by creating a key and json file for service account  
type="service_account"  
project_id="..."  
private_key_id="..."  
private_key="..."  
client_email="..."  
client_id="..."  
auth_uri="https://accounts.google.com/o/oauth2/auth"  
token_uri="https://oauth2.googleapis.com/token"  
auth_provider_x509_cert_url="https://www.googleapis.com/oauth2/v1/certs"  
client_x509_cert_url="..."  
universe_domain="googleapis.com"  
 
## Run
Locally: streamlit run main.py

On streamlit.io: 
Fork the repository so you have your own. Then sign in to Streamlit Community Cloud. Create an app based on your github repository. Then add the secrets to the app settings and reboot the app!
