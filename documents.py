import streamlit as st
import pandas as pd
import os
import glob
import pickle
from datetime import datetime
import pytz
import gcsfs

from llama_parse import LlamaParse
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient

fs = gcsfs.GCSFileSystem(token={
    'type': st.secrets.connection.gcs["type"],
    'project_id': st.secrets.connection.gcs["project_id"],
    'private_key_id': st.secrets.connection.gcs["private_key_id"],
    'private_key': st.secrets.connection.gcs["private_key"],
    'client_email': st.secrets.connection.gcs["client_email"],
    'client_id': st.secrets.connection.gcs["client_id"],
    'auth_uri': st.secrets.connection.gcs["auth_uri"],
    'token_uri': st.secrets.connection.gcs["token_uri"],
    'auth_provider_x509_cert_url': st.secrets.connection.gcs["auth_provider_x509_cert_url"],
    'client_x509_cert_url': st.secrets.connection.gcs["client_x509_cert_url"],
    'universe_domain': st.secrets.connection.gcs["universe_domain"]
})

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

# Initialize embeddings
@st.cache_resource
def get_embeddings():
    return FastEmbedEmbeddings()


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
    protected_files = fs.ls(os.path.join(gcs_bucket, gcs_protected_folder))
    customer_files = fs.ls(os.path.join(gcs_bucket, gcs_document_folder))
    filenames = [f.replace(gcs_bucket, '').replace(gcs_protected_folder, '').replace('/','')+' (protected)' for f in protected_files] + \
        [f.replace(gcs_bucket, '').replace(gcs_document_folder, '').replace('/','') for f in customer_files]
    filenames = [f for f in filenames if (f!='') & (f!=' (protected)')]
    return filenames
    

def upload_files_to_bucket(uploaded_files):
    for uploaded_file in uploaded_files:
        destination_blob_name = os.path.join(gcs_bucket, gcs_document_folder, uploaded_file.name)
        with fs.open(destination_blob_name, 'wb') as f:
            f.write(uploaded_file.getbuffer())
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
    for file in selected_files:
        path_to_delete = os.path.join(gcs_bucket, gcs_document_folder, file)
        if fs.exists(path_to_delete):
            f = fs.open(path_to_delete)
            f.fs.delete(f.path)
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
    
    # Overwrite output md file.
    parsed_output_string = "" 
    
    # Add parsed documents.
    for doc in llama_parse_documents:
        parsed_output_string = parsed_output_string + doc.text + '\n'
            
    with fs.open(os.path.join(gcs_bucket, parsed_output_file), "w") as f:
        f.write(parsed_output_string)
    
    # Split loaded documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
    docs = text_splitter.split_text(parsed_output_string)
    
    # Initialize Embeddings
    embeddings = get_embeddings()
    
    if mode=="replace":
        # Delete the existing points.
        num_points = client.count(collection_name="rag_store", exact=True).count
        if num_points > 0:
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
    path_to_parsed_data = os.path.join(gcs_bucket, parsed_data_file)
    if mode == "append":
        if fs.exists(path_to_parsed_data):
            with fs.open(path_to_parsed_data, "rb") as f:
                llama_parse_documents = llama_parse_documents + pickle.load(f)
                parsing_date = fs.modified(path_to_parsed_data)
    
    # In addition, we need to parse files which are newer than the data_file.
    protected_files = fs.ls(os.path.join(gcs_bucket, gcs_protected_folder))
    customer_files = fs.ls(os.path.join(gcs_bucket, gcs_document_folder))
    files_to_parse = [f for f in protected_files if fs.isdir(f)==False] + [f for f in customer_files if fs.isdir(f)==False]
    files_to_parse = [f for f in files_to_parse if fs.created(f) > parsing_date]

    if len(files_to_parse)==0:
        return llama_parse_documents
    
    parsing_instruction = """The provided document is a technical documentation for how to write procedures for satellites.
    The document contains many tables.
    Try to be precise while answering the questions."""
    parser = LlamaParse(api_key=llamaparse_api_key, result_type="markdown", parsing_instruction=parsing_instruction)

    for file in files_to_parse:
        # Perform the parsing step and store the result in llama_parse_documents
        if fs.exists(file):
            local_path = os.path.join("tmp", file)
            fs.download(file, local_path)
            llama_parse_documents = llama_parse_documents + parser.load_data(local_path)

    # Save the parsed data to a file
    if len(llama_parse_documents)>0:
        with fs.open(path_to_parsed_data, "wb") as f:
            pickle.dump(llama_parse_documents, f)
        
    return llama_parse_documents



def add_vertical_space(spaces=1):
    for _ in range(spaces):
        st.markdown("---") 

        
