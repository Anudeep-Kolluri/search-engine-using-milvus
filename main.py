import streamlit as st
import tiktoken
import os

tokenizer = tiktoken.get_encoding("cl100k_base")

from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core import SimpleDirectoryReader

UPLOAD_DIR = "data"

DB_NAME = "Milvus.db"
DIM = 1536

vector_store = MilvusVectorStore(
    uri = DB_NAME,
    dim = DIM,
    overwrite=True   # Need to change accordingly
)

storage_context = StorageContext.from_defaults(vector_store=vector_store)

def save_uploaded_file(uploaded_file):
    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def main():
    st.title("Retrieval Engine App")

    os.environ['OPENAI_API_KEY'] = st.sidebar.text_input("Openai API key", type = "password")
    
    # Sidebar for file uploads
    st.sidebar.title("Upload Files")
    uploaded_files = st.sidebar.file_uploader("Drop files here to retrieve from", accept_multiple_files=True)
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            save_uploaded_file(uploaded_file)

        st.toast("Files uploaded successfully")

        documents = SimpleDirectoryReader(UPLOAD_DIR).load_data()
        text = "\n".join([doc.text for doc in documents])

        tokens = tokenizer.encode(text)
        st.sidebar.write(f"**Tokens : {len(tokens)}**")

        if st.sidebar.button("Create Index"):
            st.sidebar.success("Index Created")

    # Initialize query_engine in session state if it doesn't exist
    if "query_engine" not in st.session_state:
        st.session_state.query_engine = None

    # Search bar
    search_query = st.text_input("Enter your search query")
    
    # Search button
    if st.button("Search"):
        if search_query:
            index = VectorStoreIndex.from_documents(
                documents,
                storage_context=storage_context
            )
            query_engine = index.as_retriever()
            results = query_engine.retrieve(search_query)
            print(results)
            display_results(results)
        else:
            st.warning("Please enter a search query")

def display_results(results):
    st.subheader("Top Results")
    for idx, result in enumerate(results):
        st.write(f"{idx + 1}. {result}")

if __name__ == "__main__":
    main()
