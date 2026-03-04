# -----------------------------------------------------------------
# Contains the required functions to
# - Establish connection with `Weaviate` cloud
# - Build / Get the Collection (database)
# - Apply the retriever operation

# Ahmed Ragab
# -----------------------------------------------------------------

def get_weaviate_client():
    """
    Returns:
        client: the Weaviate API required to use the database
    """
    pass


def get_collection(collection_name: str, data = None, embedding_model = None):
    """
    Builds / retrieves the collection

    Args:
        collection_name (str): the name of the collection
        data                 : the concatenated dataframes to build the database with
        embedding_model      : the model used in the vector database

    Returns:
        collection: the database
    """
    pass


def retrieve_documents(query: str, collection, n_to_return: int = 10, alpha: float = 0.7) -> list:
    """
    Retrieves the most relevant documents to the input query

    Args:
        query (str)      : the input query
        collection       : the database to retrieve from
        n_to_return (int): number of documents to return
        alpha (float)    : how much do we attend to the semantic search results

    Returns:
        retrieved_documents (list)
    """
    pass
