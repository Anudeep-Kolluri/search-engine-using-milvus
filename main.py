from pymilvus import connections, FieldSchema, DataType, CollectionSchema, Collection
from pymilvus import model
from dotenv import load_dotenv
import os


load_dotenv()

# openai_api_key = os.getenv("OPENAI_API_KEY")


host = "127.0.0.1"
port = "19530"

dim = 1536


client = connections.connect(
    host = host,
    port = port
)


# create a "collection"
def create_collection(collection_name, dim):

    from pymilvus import utility
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)

    #define schema
    col1 = FieldSchema(
        name = "id",
        dtype = DataType.VARCHAR,
        description = "ids"
        )
    
    col2 = FieldSchema(
        name = "embedding",
        dtype = DataType.FLOAT_VECTOR,
        description = "embeddings",
        dim = dim
    )

    fields = [col1, col2]

    schema = CollectionSchema(
        fields=fields,
        description="RAG search"
    )


    # collection is like a table but for vectordb
    collection = Collection(name = collection_name, schema = schema)

    # collection needs index paramters
    index_params = {
        "metric_type" : "L2",
        "index_type" : "IVF_FLAT",
        "params" : {"nlist" : 128}  #leaving at default for now. but heuristic is to put sqrt(N), where n is the # of data points (rows)
    }

    collection.create_index(
        field_name="embedding",
        index_params = index_params
        )
    
    return collection


# create embeddings of the data
embedding_model = model.dense.OpenAIEmbeddingFunction(
    model_name = "text-embedding-ada-002",
)



# table/collection is created
collection = create_collection("RAG", dim)
