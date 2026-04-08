# ------------------------------------------------------------
# Prepare the database & record the relevant doc in it
# ------------------------------------------------------------

import os
import weaviate
from weaviate.classes.init import Auth, AdditionalConfig, Timeout
from weaviate.classes.config import DataType, Configure, Property
from tqdm.auto import tqdm

def get_weaviate_client():
    """Connect to the weaviate cloud"""
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url = os.getenv("WEAVIATE_URL"),
        auth_credentials = Auth.api_key(os.getenv("WEAVIATE_API_KEY")),
        additional_config = AdditionalConfig(
            timeout = Timeout(init = 30, query = 60, insert = 120)
        )
    )

    return client


def create_collection(client, collection_name: str):
    """
    Building a Collection for the Eval Data
    """
    # delete the collections to rebuild it using the new `embedding_model`
    # --> That is to ensure the dimensions are matched between the objects in the collection & the query
    # --> Also to calc the time for embedding the chunks using the `embedding_model`
    if client.collections.exists(collection_name):
        client.collections.delete(collection_name)

    collection = client.collections.create(
        name = collection_name,
        vectorizer_config = Configure.Vectorizer.none(),
        properties = [
            Property(name = "chunk_text", data_type = DataType.TEXT),
            Property(name = "chunk_topic", data_type = DataType.TEXT),
            Property(name = "chunk_id", data_type = DataType.INT)
        ]
    )

    return collection


def prepare_collection(client, collection_name: str, embedding_model, chunks):
    """Insert the relevant documents into the collection"""
    
    collection = create_collection(client, collection_name)

    with collection.batch.fixed_size(batch_size = 50, concurrent_requests = 1) as batch:
        for chunk in tqdm(chunks, total = len(chunks)):
            vectorized_chunk = embedding_model.embed_query(chunk["chunk"])

            batch.add_object(
                properties = {
                    "chunk_text" : chunk["chunk"],
                    "chunk_id"   : chunk["id"],
                    "chunk_topic": chunk["main_topic"]
                },
                vector = vectorized_chunk
            )
    
    return collection
