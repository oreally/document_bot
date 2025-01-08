import streamlit as st
import pandas as pd
import os
import glob
import pickle
from datetime import datetime
import pytz

from google.cloud import storage

from llama_parse import LlamaParse
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient

llamaparse_api_key = st.secrets["LLAMA_CLOUD_API_KEY"]
qdrant_url = st.secrets["QDRANT_URL"]
qdrant_api_key = st.secrets["QDRANT_API_KEY"]
gcs_bucket = st.secrets["GCS_BUCKET"]
gcs_protected_folder = st.secrets["GCS_PROTECTED_FOLDER"]
gcs_document_folder = st.secrets["GCS_DESTINATION_FOLDER"]
parsed_data_file = st.secrets["PARSED_DATA_FILE"]
parsed_output_file = st.secrets["PARSED_OUTPUT_FILE"]
project_id = st.secrets.connection.gcs["project_id"]
utc=pytz.UTC

# Initializing Qdrant client
@st.cache_resource
def load_qdrant_client():
    client = QdrantClient(api_key=qdrant_api_key, url=qdrant_url)
    return client

client = load_qdrant_client()


def document_manager():
    # Document upload
    st.subheader('Upload documents')
    if "documents" not in st.session_state.keys():
        st.session_state.documents = list_documents_in_bucket()
    with st.form("my-form", clear_on_submit=True):
        uploaded_files = st.file_uploader("Choose files", type=["pdf", "docx", "html", 'md', 'txt'], key='uploader',
                     accept_multiple_files=True, label_visibility="hidden")
        submitted = st.form_submit_button("UPLOAD!", use_container_width=True)

    if submitted and uploaded_files is not None:
        with st.spinner("Uploading..."):
            st.session_state.documents = upload_files_to_bucket(uploaded_files)

    # Display document list and delete functionality
    document_editor()

    if st.button('Update vector store', use_container_width=True):
        with st.spinner("Parsing files..."):
            create_vector_database()
        st.write("Successfully parsed all files.")
        
    # Adjust the number of spaces as needed
    add_vertical_space(1)  
    st.markdown('''
        ## About
                    
        The Satellite Chatbot uses Google Cloud Storage, Llamaparse, Qdrant and the mixtral-8x7b-32768 LLM from Groq.
             ''',
        unsafe_allow_html=True)  


def list_documents_in_bucket():
    storage_client = storage.Client(project=project_id)
    blobs_1 = storage_client.list_blobs(gcs_bucket, prefix=gcs_protected_folder)
    blobs_2 = storage_client.list_blobs(gcs_bucket, prefix=gcs_document_folder)
    filenames = [blob.name.replace(gcs_protected_folder, '').replace('/','')+' (protected)' for blob in blobs_1] + [blob.name.replace(gcs_document_folder, '').replace('/','') for blob in blobs_2]
    filenames = [f for f in filenames if (f!='') & (f!=' (protected)')]
    return filenames
    

def upload_files_to_bucket(uploaded_files):
    storage_client = storage.Client(project=project_id)
    bucket = storage_client.bucket(gcs_bucket)
    for uploaded_file in uploaded_files:
        destination_blob_name = os.path.join(gcs_document_folder, uploaded_file.name)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_file(uploaded_file)
    return list_documents_in_bucket()


def document_editor():
    if "documents" in st.session_state and len(st.session_state.documents)>0:
        if ("df" not in st.session_state.keys()) or (st.session_state["df"].shape[0] < len(st.session_state.documents)):
            st.session_state["df"] = pd.concat((
                pd.Series([False]*len(st.session_state.documents), name='Delete'),
                pd.Series(st.session_state.documents, name='Document')), axis=1)
        df_editor = st.data_editor(
            st.session_state["df"], hide_index=True, disabled=['Document'],
            key="df_editor", on_change=df_on_change, use_container_width=True
        )
    
def df_on_change():
    editor = st.session_state["df_editor"]
    df = st.session_state["df"]
    files_to_delete=[]
    for index, updates in editor["edited_rows"].items():
        if updates['Delete']:
            files_to_delete.append(df[df.index==index].Document.values[0])
    st.session_state.documents = delete_documents_from_bucket(files_to_delete)
    df = pd.concat((
        pd.Series([False]*len(st.session_state.documents), name='Delete'),
        pd.Series(st.session_state.documents, name='Document')), axis=1)
    st.session_state["df"] = df
    
    
def delete_documents_from_bucket(selected_files):
    storage_client = storage.Client(project=project_id)
    bucket = storage_client.bucket(gcs_bucket)
    
    for file in selected_files:
        path_to_delete = os.path.join(gcs_document_folder, file)
        blob = bucket.blob(path_to_delete)
        if blob.exists():
            # Optional: set a generation-match precondition to avoid potential race conditions
            # and data corruptions. The request to delete is aborted if the object's
            # generation number does not match your precondition.
            blob.reload()  # Fetch blob metadata to use in generation_match_precondition.
            generation_match_precondition = blob.generation
            blob.delete(if_generation_match=generation_match_precondition)
    return list_documents_in_bucket()


def create_vector_database(mode="replace"):
    """
    Creates a vector database using document loaders and embeddings.

    This function loads urls,
    splits the loaded documents into chunks, transforms them into embeddings using OllamaEmbeddings,
    and finally persists the embeddings into a Chroma vector database.

    """
    # Call the function to either load or parse the data
    llama_parse_documents = load_or_parse_data(mode=mode)
    
    # Load output md file.
    storage_client = storage.Client(project=project_id)
    bucket = storage_client.bucket(gcs_bucket)
    parsed_output = bucket.blob(parsed_output_file)
    parsed_output_string = "" # We overwrite it.
    
    # Add parsed documents.
    for doc in llama_parse_documents:
        parsed_output_string = parsed_output_string + doc.text + '\n'

    if len(parsed_output_string)>0:
        parsed_output.upload_from_string(parsed_output_string)
    
    # Split loaded documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
    docs = text_splitter.split_text(parsed_output_string)
    
    # Initialize Embeddings
    embeddings = FastEmbedEmbeddings()
    
    if mode=="replace":
        # Delete the existing points.
        num_points = client.count(collection_name="rag_store", exact=True).count
        points = client.scroll(collection_name="rag_store", limit=num_points)
        ids = [p.id for p in points[0]]
        vectorstore = Qdrant(client=client, embeddings=embeddings, collection_name="rag_store")
        vectorstore.delete(ids=ids)
        
    # Create and persist a Chroma vector database from the chunked documents.
    qdrant = Qdrant.from_texts(
        texts=docs,
        embedding=embeddings,
        url=qdrant_url,
        collection_name="rag_store",
        api_key=qdrant_api_key
    )

    st.write('Vector DB created successfully !')

    
def load_or_parse_data(mode="replace"):
    llama_parse_documents = []
    parsing_date = utc.localize(datetime.strptime('1970-01-01', '%Y-%m-%d'))
    
    # Load already parsed data, if it exists.
    storage_client = storage.Client(project=project_id)
    bucket = storage_client.bucket(gcs_bucket)
    parsed_data = bucket.blob(parsed_data_file)
    if mode == "append":
        parsed_data_info = bucket.get_blob(parsed_data_file)
        if parsed_data.exists():
            pickle_in = parsed_data.download_as_string()
            llama_parse_documents = llama_parse_documents + pickle.loads(pickle_in)
            parsing_date = parsed_data_info.updated
    
    # In addition, we need to parse files which are newer than the data_file.
    blobs_1 = storage_client.list_blobs(gcs_bucket, prefix=gcs_protected_folder)
    blobs_2 = storage_client.list_blobs(gcs_bucket, prefix=gcs_document_folder)
    files_to_parse = [blob for blob in blobs_1 if blob.updated > parsing_date] + [blob for blob in blobs_2 if blob.updated > parsing_date]
    files_to_parse = [f for f in files_to_parse if (f.name!=gcs_protected_folder+'/') & (f.name!=gcs_document_folder+'/')]

    if len(files_to_parse)==0:
        return llama_parse_documents
    
    parsing_instruction = """The provided document is a technical documentation for how to write procedures for satellites.
    The document contains many tables.
    Try to be precise while answering the questions."""
    parser = LlamaParse(api_key=llamaparse_api_key, result_type="markdown", parsing_instruction=parsing_instruction)

    for file in files_to_parse:
        # Perform the parsing step and store the result in llama_parse_documents
        destination_file_name = os.path.join("/tmp", file.name.split('/')[-1])
        file.download_to_filename(destination_file_name)
        llama_parse_documents = llama_parse_documents + parser.load_data(destination_file_name)

    # Save the parsed data to a file
    if len(llama_parse_documents)>0:
        pickle_out = pickle.dumps(llama_parse_documents)
        parsed_data.upload_from_string(pickle_out)
        
    return llama_parse_documents



def add_vertical_space(spaces=1):
    for _ in range(spaces):
        st.markdown("---") 

        
