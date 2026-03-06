# -----------------------------------------------------------------
# Contains the required functions to
# - Establish connection with `Weaviate` cloud
# - Build / Get the Collection (database)
# - Apply the retriever operation

# Ahmed Ragab
# -----------------------------------------------------------------
import weaviate
import os
from weaviate.classes.init import Auth, AdditionalConfig, Timeout
from weaviate.classes.config import Configure, Property, DataType
from tqdm.auto import tqdm

def get_weaviate_client():
    """
    Returns:
        client: the Weaviate API required to use the database
    """
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url = os.getenv("WEAVIATE_URL"),
        auth_credentials = Auth.api_key(api_key = os.getenv("WEAVIATE_API_KEY")),
        additional_config = AdditionalConfig(
            timeout = Timeout(init = 30, query = 60, insert = 120)
        )
    )

    return client


def build_collection(client, collection_name: str, data = None, embedding_model = None):
    """
    Builds / retrieves collection

    Args:
        client               : the weaviate client
        collection_name (str): the name of the collection
        data                 : the concatenated dataframes to build the database with
        embedding_model      : the model used in the vector database

    Returns:
        collection: the database
    """
    # retrieve if exists
    if client.collections.exists(collection_name):
        collection = client.collections.get(collection_name)
        print(">> Collection Exists")
    
    # build if not exists
    else:
        collection = client.collections.create(
            name = collection_name,
            vector_config = Configure.Vectorizer.none(),
            properties = [
                Property(name = 'job_document', data_type = DataType.TEXT),
                Property(name = 'year', data_type = DataType.INT)
            ]
        )
        print(">> Collection Created")

    # add data
    if data is not None:
        batch_size = 120
        total_rows = len(data)
        with collection.batch.dynamic() as batch:
            for i in tqdm(range(0, total_rows, batch_size), desc = "Uploading to Weaviate"):
                batch_df = data.iloc[i : i + batch_size]
                batch_vectors = embedding_model.embed_documents(batch_df['job_document'].tolist())

                for idx, row in enumerate(batch_df.itertuples(index = False)):
                    batch.add_object(
                        properties = {
                            'job_document' : row.job_document,
                            'year'         : int(row.year)
                        },
                        vector = batch_vectors[idx]
                    )

    return collection


def load_collection(client, collection_name: str):
    """
    Retrieves collection with data
    Args:
        client               : the weaviate client
        collection_name (str): the name of the collection
    """
    if client.collections.exists(collection_name):
        return client.collections.get(collection_name)
    else:
        raise ValueError(f"Collection with {collection_name} is not found")
    

def retrieve_documents(query: str, collection, embedding_model, n_to_return: int = 10, alpha: float = 0.7) -> list:
    """
    Retrieves the most relevant documents to the input query

    Args:
        query (str)      : the input query
        collection       : the database to retrieve from
        embedding_model  : model used to embed the query
        n_to_return (int): number of documents to return
        alpha (float)    : how much do we attend to the semantic search results

    Returns:
        retrieved_documents (list) sorted by year
    """
    query_embedded = embedding_model.embed_query(query)
    retrieved = collection.query.hybrid(
        query = query,
        vector = query_embedded,
        limit = n_to_return,
        alpha = alpha
    ).objects


    # sort by year
    retrieved_sorted = sorted(
        retrieved,
        key = lambda x : x.properties.get('year', 0),
        reverse = True
    )

    return retrieved_sorted



    



