# ------------------------------------------------------------
# Load & Evaluate the Different RAG Components
# ------------------------------------------------------------

from langchain_huggingface import HuggingFaceEmbeddings
import src.config as CFG
import time
import torch
from sentence_transformers import CrossEncoder
# from FlagEmbedding  import FlagReranker
# ------------------------------------------------------------------------------------------------
# Loading Models
def load_embedding_model(model_name: str, device = CFG.DEVICE):
    """Load the required embedding model using `HuggingFaceEmbeddings`"""
    if model_name == "bge":
        return HuggingFaceEmbeddings(
            model_name = CFG.BGE_EMBEDDING_MODEL_NAME,
            model_kwargs = {"device" : device, "trust_remote_code": True},
            encode_kwargs = {"batch_size" : CFG.EMBEDDING_BATCH_SIZE}
        )

    elif model_name == "nomic":
        return HuggingFaceEmbeddings(
            model_name = CFG.NOMIC_EMBEDDING_MODEL_NAME,
            model_kwargs = {"device": device, "trust_remote_code": True},
            encode_kwargs = {"batch_size" : CFG.EMBEDDING_BATCH_SIZE}
        )

    
    elif model_name == "Qwen2":
        raise ValueError("Model is very big to either download or use")
    else:
        raise ValueError("Model Name is not Valid")
    

def load_reranker(model_name: str, device = CFG.DEVICE):
    if model_name == "minilm":
        return CrossEncoder(CFG.MINILM_RERANKER_MODEL_NAME)
    elif model_name == "mixedbread":
        return CrossEncoder(CFG.MIXEDBREAD_RERANKER_MODEL_NAME)
    # elif model_name == "bge":
    #     return FlagReranker(CFG.BGE_RERANKER_MODEL_NAME, use_fp16 = True)
    
# ------------------------------------------------------------------------------------------------
# Embedding & Retreiving
def embed_chunks(model, chunks: list, device = CFG.DEVICE):
    """
    Args:
        model (HF model): the embedding model to use in embedding
        chunks (list)   : the list of chunks to embed

    Returns:
        embeddings (list)  : the list of embeddings 
    """

    if str(device) == 'cuda':
        embeddings = model.embed_documents(chunks)
        torch.cuda.synchronize()

    else:
        embeddings = []
        for i in range(0, len(chunks), CFG.EMBEDDING_BATCH_SIZE):
            batch = chunks[i : i + CFG.EMBEDDING_BATCH_SIZE]
            batch_embeddings = model.embed_documents(batch)
            embeddings.extend(batch_embeddings)
        
    return embeddings  
    
    
def retreive(query: str, embedding_model, collection, n_to_return: int = 10, alpha: float = 0.7):
    """The Retrieval Operation"""

    embedded_query = embedding_model.embed_query(query)
    retreived_objects = collection.query.hybrid(
        query = query,
        vector = embedded_query,
        limit = n_to_return,
        alpha = alpha
    ).objects

    return retreived_objects


def rerank(query, cross_encoder, retreived_objects, n_to_return = 5, model_name: str = 'minilm'):
    model_inputs = [[query, obj.properties.get("chunk_text")] for obj in retreived_objects]

    if model_name == "bge":
        scores = cross_encoder.compute_score(model_inputs)
    else:
        scores = cross_encoder.predict(model_inputs)

    chunks_with_scores = list(zip(retreived_objects, scores))
    sorted_by_scores = sorted(chunks_with_scores, key = lambda x : x[1], reverse = True, )
    
    reranked = [obj[0] for obj in sorted_by_scores] # get the chunks only ()
    return reranked[:n_to_return]
# ------------------------------------------------------------------------------------------------

# Calc Metrics
def calc_recall_at_k(relevant_chunks_ids, retreived_chunks_ids):
    total_relevant = len(relevant_chunks_ids)
    n_retreived_relevant = 0

    for chunk_id in retreived_chunks_ids:
        if chunk_id in relevant_chunks_ids:
            n_retreived_relevant += 1

    recall = n_retreived_relevant / total_relevant
    return recall




def calc_precision_at_k(relevant_chunks_ids, retreived_chunks_ids):
    total_retreived = len(retreived_chunks_ids)
    n_retreived_relevant = 0

    for chunk_id in retreived_chunks_ids:
        if chunk_id in relevant_chunks_ids:
            n_retreived_relevant += 1

    precision = n_retreived_relevant / total_retreived
    return precision


def calc_mrr(job_topic: str, retreived_chunks_topics: list[str]):
    """Calc the Mean Reciporcal Rank based on the topic"""
    rank = 0

    for idx, topic in enumerate(retreived_chunks_topics, start = 1):
        print(job_topic)
        print(topic)
        if topic == job_topic:
            rank = 1 / idx
            break
    
    return rank

# ------------------------------------------------------------------------------------------------

# Evaluate
def get_embedding_time(model, chunks: list, repeats: int, device = CFG.DEVICE):
    """
    Args:
        model (HF model): the embedding model to use in embedding
        chunks (list)   : the list of chunks to embed
        repeats (int)   : number of times to embed the chunks [used to calc avg time]

    Returns:
        dict of:
            embeddings (list)  : the list of embeddings 
            avg_time   (float) : the average time for embedding the chunks for (repeats) times
    """
    # get embeddings [before calc time to warm the GPU]
    _ = embed_chunks(model, chunks, device) 

    total_time = 0
    for _ in range(repeats):
        start_time = time.perf_counter()    
        _ = embed_chunks(model, chunks, device) 
        total_time += time.perf_counter() - start_time

    avg_time = total_time / repeats

    return avg_time


def evaluate_retreival_operation(
        eval_data: list, 
        embedding_model, 
        reranker,
        collection, 
        reranker_name: str = "minilm",
        n_to_retreive: int = 10,
        n_to_return: int = 5, 
        alpha: float = 0.7
    ):
    """Evaluating the Retreival"""

    total_recall_at_k_score = 0
    total_precision_at_k_score = 0
    total_mrr = 0

    for sample in eval_data:
        # get query
        query = sample["original_job_description"]

        # retreive & calc recall
        retreived_objects = retreive(
            query = query,
            embedding_model = embedding_model,
            collection = collection,
            n_to_return = n_to_retreive,
            alpha = alpha
        )

        recall_at_k = calc_recall_at_k(
            relevant_chunks_ids = sample["relevant_chunks_ids"],
            retreived_chunks_ids = [obj.properties.get("chunk_id") for obj in retreived_objects]
        )
        total_recall_at_k_score += recall_at_k

        # rerank & calc precision
        reranked = rerank(
            query = query,
            cross_encoder = reranker,
            retreived_objects = retreived_objects,
            n_to_return = n_to_return,
            model_name = reranker_name
        )

        precision_at_k = calc_precision_at_k(
            relevant_chunks_ids = sample["relevant_chunks_ids"],
            retreived_chunks_ids = [obj.properties.get("chunk_id") for obj in reranked]
        )
        total_precision_at_k_score += precision_at_k


        # calc mrr
        mrr = calc_mrr(
            job_topic = sample["main_topic"],
            retreived_chunks_topics = [obj.properties.get("chunk_topic") for obj in reranked]
        )
        total_mrr += mrr
    
    avg_recall_at_k_score = total_recall_at_k_score / len(eval_data)
    avg_precision_at_k_score = total_precision_at_k_score / len(eval_data)
    avg_mrr = total_mrr / len(eval_data)

    return {
        "avg_recall_at_k_score"       : avg_recall_at_k_score,
        "avg_precision_at_k_score"    : avg_precision_at_k_score,
        "average_mean_reciporcal_rank": avg_mrr
    }