
#-------------------------- COMMAND LINE ARGUMENTS -----------------------
import argparse

parser = argparse.ArgumentParser(description='RAG search engine using Milvus')
parser.add_argument('-o', '--overwrite', action='store_true', help='Overwrite the current database')
parser.add_argument('-r', '--reload', action='store_true', help="Reload data folder and re-read the embeddings [EXPENSIVE]")

args = parser.parse_args()

#------------------------ HANDING VECTOR STORE ---------------------------

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core import StorageContext

DB_NAME = "test.db"

embedding_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
# embedding_dim = embedding_model.embedding_dimension


vector_store = MilvusVectorStore(
    uri = DB_NAME,
    dim = 1536,
    overwrite=args.overwrite
    )

storage_context = StorageContext.from_defaults(vector_store = vector_store)

#---------------------- GETTING THE EMBEDDINGS ---------------------------
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex, StorageContext

if args.overwrite or args.reload:
    documents = SimpleDirectoryReader("./test_data").load_data()
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context
    )
    print("Data Loaded successfully into the Database")
else:
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store
    )

#----------------------- QUERY ------------------------
# query_engine = index.as_query_engine(llm = None)
# response = query_engine.query("What model are you using?", llm = None)

# print(response)


query = "What are transformers"
query_engine = index.as_retriever()
results = query_engine.retrieve(query)

print(results)