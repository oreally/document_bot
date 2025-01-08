import os
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate  
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from qdrant_client import QdrantClient
from langchain.chains import RetrievalQA

import streamlit as st

llamaparse_api_key = st.secrets["LLAMA_CLOUD_API_KEY"]
qdrant_url = st.secrets["QDRANT_URL"]
qdrant_api_key = st.secrets["QDRANT_API_KEY"]
groq_api_key = st.secrets["GROQ_API_KEY"]

# Custom prompt template for QA retrieval
custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else. 
Helpful answer:
"""

# Initializing Qdrant client
@st.cache
def load_qdrant_client():
    client = QdrantClient(api_key=qdrant_api_key, url=qdrant_url)
    return client

client = load_qdrant_client()

# Initializing chat model: 
@st.cache
def load_chat_model():
    model = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768")
    return model

chat_model = load_chat_model()


def chatbot():
    # Store LLM generated responses
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": "Hi, how can I help you?", "sources": None}]

    # Display chat messages
    chat_history = st.container(height=650)

    for message in st.session_state.messages:
        with chat_history.chat_message(message["role"]):
            st.write(message["content"])
            if message["sources"]:
                for source in message["sources"]:
                    with st.expander(source['name'], icon=":material/book_4_spark:"):
                        st.markdown(source['content'], unsafe_allow_html=True)

    # User-provided prompt
    if query := st.chat_input("Enter a question", key="user_input"):
        st.session_state.messages.append({"role": "user", "content": query, "sources": None})
        with chat_history.chat_message("user"):
            st.write(query)

    # Generate a new response if last message is not from assistant
    if (st.session_state.messages[-1]["role"] != "assistant") and (query is not None):
        with chat_history.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response, sources = qa_bot(query)
                st.write(response) 
                if sources:
                    for source in sources:
                        with st.expander(source['name'], icon=":material/book_4_spark:"):
                            st.markdown(source['content'], unsafe_allow_html=True)
                    
        message = {"role": "assistant", "content": response, "sources": sources}
        st.session_state.messages.append(message)
        

def qa_bot(query):
    """
    Function to set up QA bot
    """
    embeddings = FastEmbedEmbeddings()
    vectorstore = Qdrant(client=client, embeddings=embeddings, collection_name="rag_store")
    llm = chat_model
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, vectorstore)
    response = qa({'query': query})
    answer = response["result"]
    source_documents = response["source_documents"]
    links = []
    if source_documents:
        # Remove duplicate sources.
        source_documents = list( dict.fromkeys([s.page_content for s in source_documents]) )
        for source_idx, source_doc in enumerate(source_documents):
            source_name = f"source-{source_idx + 1}"
            links.append({"name": source_name, "content": source_doc})
            # Jumping on the same page works, but there is a problem with space, so we don't do the following:
            # st.session_state.sources.append({"name": source_name, "content": source_doc.page_content})
            # links.append(f'''<a href='#{source_name}'>{source_name}</a>''')
            # Jumping to a new page clears the cache, so the following does not work:
            # links.append(f'''[{source_name}]({app_path}/source_page#{source_name})''')
            # st.markdown(f'''Sources: {', '.join(sources)}''', unsafe_allow_html=True)
    return answer, links


def set_custom_prompt():
    """
    Function to set custom prompt template for QA retrieval
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=["context", "question"])
    return prompt


def retrieval_qa_chain(llm, prompt, vectorstore):
    """
    Function to create retrieval QA chain
    """
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    return qa_chain
