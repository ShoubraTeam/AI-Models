# ---------------------------------------------------------------------------
# Contains the required class to
# - Initiate the enhancer
# - enhancing the given job desc
# - detect the skills
# ---------------------------------------------------------------------------



from src.utils.connect_to_groq import get_groq_client, enhance_old_job, has_skills
from src.vector_database import get_weaviate_client, load_collection, retrieve
from src.utils.prompts import get_enhancement_prompt, get_detection_prompt
import src.utils.config as CFG
from src.utils.format_jobs import format_job, format_for_retriever

class Enhancer:
    """
    General class for Job Description Enhancement / Suggestion
    """
    def __init__(self, enhancement_model: str, detection_model: str, collection_name: str, model_provider = 'groq'):
        # attbs
        self.enhancement_model = enhancement_model
        self.detection_model = detection_model
        self.model_provider = model_provider
        self.collection_name = collection_name

        # setup client
        self.load_client()
        
        # setup weaviate vector DB
        self.weaviate_client = get_weaviate_client()
        self.collection = load_collection(self.weaviate_client, self.collection_name)


    def load_client(self):
        if self.model_provider == "groq":
            self.client = get_groq_client()
    # -------------------------------------------------------------------------------------
    def detect_skills(self, job_desc: str) -> bool:
        """
        Determine if the original job description contains skills or not

        Args:
            job_desc: the original job description
        """
        system_prompt = get_detection_prompt()
        response = has_skills(
            client = self.client,
            query = job_desc,
            model_name = self.detection_model,
            system_prompt = system_prompt,
            temperature = 0 
        )
        
        return "yes" in response.lower()
    # -------------------------------------------------------------------------------------
    def enhance_old_desc(self, job_info: dict, use_rag: bool, stream = False, **kwargs) -> str:
        """
        Enhances an existing job description using an LLM

        Args:
            job_info: dictionary contains the job info (title - desc - skills - categories - year)
            use_rag: whether to use RAG or not
            stream: If True, returns a streaming response. Defaults to False.
            **kwargs: Additional parameters passed directly to the model API
                (e.g., temperature, max_tokens).
        """   
        system_prompt = get_enhancement_prompt(use_rag = use_rag)

        if use_rag:
            formatted = format_for_retriever(
                job_title = job_info["title"],
                job_desc = job_info["description"]
            )
            retrieved = retrieve(
                retriever_query = formatted,
                embedding_model = CFG.EMBEDDING_MODEL,
                cross_encoder = CFG.RERANKER,
                collection = self.collection,
                n_to_return = 5 
            )

            retrieved_docs_text = [doc[0].properties.get("job_document") for doc in retrieved]

        formatted_job = format_job(
            job_info = job_info,
            use_rag = use_rag,
            retrieved_documents = retrieved_docs_text
        )

        response = enhance_old_job(
            client = self.client,
            query = formatted_job,
            model_name = self.enhancement_model,
            system_prompt = system_prompt,
            stream = stream,
            **kwargs
        )

        return response, formatted_job
    

    def close_db(self):
        self.weaviate_client.close()