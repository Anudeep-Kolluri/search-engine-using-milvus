#---------------------- OPENAI PART --------------------------
import openai
from dotenv import load_dotenv
import os

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


#----------------------- LLAMA INDEX PART (EMBEDDING) ------------------------
# from llama_index.core import SimpleDirectoryReader
# from llama_index.core import VectorStoreIndex

# documents = SimpleDirectoryReader("./data").load_data()

# vector_index = VectorStoreIndex.from_documents(documents, show_progress = True)
# vector_index.as_query_engine()


from llama_index.core import Document
from llama_index.core import VectorStoreIndex

doc = Document(text="Hi, my name is anudeep and I am really excited to learn all these new technologies!")


#--------------------- MILVUS PART (STORING EMBEDDINGS) --------------------------
from llama_index.core import VectorStoreIndex, StorageContext

from llama_index.vector_stores.milvus import MilvusVectorStore

vector_store = MilvusVectorStore(dim=1536, overwrite=True)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    [doc], storage_context=storage_context
)

#------------------------ QUERY PART ---------------------------------
query_engine = index.as_query_engine()
response = query_engine.query("What is your name?")

print(response)