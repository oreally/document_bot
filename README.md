# RAG-Chat-with-Documents
Chainlit app for RAG chat with documents Parsing PDF documents using LlamaParse, Qdrant, and the Groq model.

Original source: https://github.com/divakarkumarp/RAG-Chat-with-Documents/tree/main

However, the same code seems to be used all over the place, e.g. https://huggingface.co/ThisIs-Developer/Llama-2-GGML-Medical-Chatbot

I don't know who first created the app.
 
## Overview:
Software And Tools Requirements

1. [LlamaParse](https://cloud.llamaindex.ai/)
2. [Qdrant](https://cloud.qdrant.io/)
3. [Groq](https://groq.com/)
4. [Langchain🦜](https://www.langchain.com/)
5. [Langchain Smith 🦜](https://smith.langchain.com/o/32390bae-a13d-5a53-b61b-501e3f39e496/projects/p/7e7575b9-5a88-46e5-b7d1-819569ebb004?timeModel=%7B%22duration%22%3A%227d%22%7D&tab=0)

## Prepare
1. Create Llama-Cloud API-KEY
2. Create Qdrant cluster and API-KEY
3. Create Groq API-KEY
4. Store API-KEYs and Qdrant cluster endpoint URL in .env file in RAG-Chat-with-Documents folder.
 
## Run
On Linux:  
Install python 3.9, virtualenv, source  
1. python3.9 -m venv .venv && source .venv/bin/activate
2. cd RAG-Chat-with-Documents/
3. pip install -r requirements.txt
4. python3.9 ingest.py
5. chainlit run app.py

On Windows:  
Install python 3.9, virtualenv, source  
1. Open Anaconda and create a new virtual environment using python 3.9. 
2. Open the environment in a Terminal (click on the green Run button next to the environment name in anaconda).
3. cd RAG-Chat-with-Documents/
4. python3.9 -m pip install -r requirements.txt
5. python3.9 ingest.py
6. chainlit run app.py

