

from helpers.config import JOB_DESCRIPTION_ENHANCEMENT_COLLECTION_V1
from helpers.config import get_settings
from weaviate import WeaviateClient

from weaviate.classes.config import Configure, Property, DataType
from tqdm.auto import tqdm
import pandas as pd

from models.enums import SuccessEnum, ErrorsEnum
from helpers.functional import print_success_message


class WeaviateController:
    """
    Controlling the logic of:-
        - Build Collection
        - Retrieving & Reranking
    
    Args:
        agents: embedding model to retrieve relevant documents - documents reranker model
        client: Weaviate Client
    """
    # ---------------------------------------- Setup -------------------------------------------
    def __init__(self, agents, client: WeaviateClient) -> None:
        if agents is not None:
            self.embedder = agents["RAG_embedder"]
            self.reranker = agents["RAG_reranker"]
        
        else:
            self.embedder = None
            self.reranker = None
        self.client = client
        self.documents_path = get_settings().JOB_DESCRIPTION_ENHANCEMENT_DATA_PATH
    

    def get_collection(self):
        """Returns Collection or None if collecion not exists"""
        if not self.client.collections.exists(JOB_DESCRIPTION_ENHANCEMENT_COLLECTION_V1):
            raise None

        collection = self.client.collections.get(JOB_DESCRIPTION_ENHANCEMENT_COLLECTION_V1)
        return collection
        
        
    def build_collection(self):
        collection = self.client.collections.create(
            name              = JOB_DESCRIPTION_ENHANCEMENT_COLLECTION_V1,
            vectorizer_config = Configure.Vectorizer.none(),
            properties = [
                Property(name = "job_document", data_type = DataType.TEXT),
                Property(name = "year", data_type = DataType.INT)
            ] 
        )

        return collection
    
    def fill_collection(self, collection):
        try:
            data = pd.read_parquet(self.documents_path)
        except:
            raise

        if data is None or data.empty:
            raise
        
        with collection.batch.fixed_size(batch_size = 50, concurrent_requests = 4) as batch:
             for row in tqdm(data.itertuples(index = False), total = len(data), desc = "Uploading"):
                 batch.add_object(
                    properties = {
                        'job_document': row.job_document,
                        'year': int(row.year)
                    },
                    vector = row.embeddings
                )
        

        return collection
    
    # ---------------------------------------- Retrieve & Rerank -------------------------------------------
    def retrieve_documents(self, query: str, collection, n_to_return: int = 50, alpha: float = 0.7) -> list:
        """
        retrieves the most relevant documents to the input query

        Args:
            query (str)      : the input query
            collection       : the database to retrieve from
            n_to_return (int): number of documents to return
            alpha (float)    : how much do we attend to the semantic search results

        Returns:
            retrieved_documents (list) sorted by year
        """
        query_embedded = self.embedder.embed_query(query)
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

        return retrieved_sorted # obj (document_job, year)

    def rerank_documents(self, query: str, documents_objects: list, n_to_return: 10):
        """
        Reranking the retrieved documents using a cross_encoder to guarantee that the LLM receives the most relevant context possible.

        Args:
            query (str)                                : the rerank query
            documents_objects (list)                   : retrieved documents objs to rerank
            n_to_return (int)                          : number of documents to return after reranking

        Returns:
            reranked_documents (list)
        """
        model_inputs = [[query, doc.properties.get("job_document")] for doc in documents_objects]  

        scores = self.reranker.predict(model_inputs)  

        docs_with_scores = list(zip(documents_objects, scores))

        reranked_documents = sorted(docs_with_scores, key = lambda x : x[1], reverse = True)
        
        return reranked_documents[:n_to_return]  # (obj, score), --> obj (job_doc, year)


    def retrieve(
        self,
        collection,
        retriever_query: str,
        reranker_query: str = None,
        n_to_return: int = 10,
        alpha: float = 0.7,
    ):
        """
        retrieve the most (n_to_return) relevant documents from the collection of documents given

        Args:
            retriever_query (str)      : the retriever query (original input query)
            reranker_query (str): query used in reranking. If None -> use retriever_query
            n_to_return (int): number of documents to return
            alpha (float)    : how much do we attend to the semantic search results

        Retunrs:
            documents (list of retrieved documents_text)
        """
        retrieved_documents = self.retrieve_documents(
            query = retriever_query,
            collection = collection,
            n_to_return = 50,
            alpha = alpha
        )


        if reranker_query is None:
            reranker_query = retriever_query

        reranked_documents = self.rerank_documents(
            query = reranker_query,
            documents_objects = retrieved_documents,
            n_to_return = n_to_return
        ) 

        documents = [doc[0].properties.get("job_document") for doc in reranked_documents]

        return documents  # job_doc --> job detais


    