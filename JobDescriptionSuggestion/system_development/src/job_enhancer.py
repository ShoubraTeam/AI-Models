# ---------------------------------------------------------------------------
# Contains the required class to
# - Initiate the enhancer
# - enhancing the given job desc
# - detect the skills
# ---------------------------------------------------------------------------


import ast
from src.utils.connect_to_groq import get_groq_client, enhance_old_job, has_tools, extract_tools
from src.vector_database import get_weaviate_client, load_collection, retrieve
from src.utils.prompts import get_enhancement_prompt, get_detection_prompt, get_tools_prompt
import src.utils.config as CFG
from src.utils.format_jobs import format_for_enhancement, format_for_retriever, format_retrieved_docs

class Enhancer:
    """
    General class for Job Description Enhancement / Suggestion
    """
    def __init__(self, enhancement_model: str, detection_model: str, skills_extractor: str, collection_name: str, model_provider = 'groq'):
        # attbs
        self.enhancement_model = enhancement_model
        self.detection_model = detection_model
        self.skills_extractor = skills_extractor
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
    def detect_tools(self, job_desc: str) -> bool:
        """
        Determine if the original job description contains tools or not

        Args:
            job_desc: the original job description
        """
        system_prompt = get_detection_prompt()
        response = has_tools(
            client = self.client,
            query = job_desc,
            model_name = self.detection_model,
            system_prompt = system_prompt,
            temperature = 0,
            max_tokens = 1 
        )
        
        if response.lower() == 'yes' or response.lower() == "no":
            return "yes" in response.lower()
        else:
            return self.detect_tools(job_desc)
    # -------------------------------------------------------------------------------------
    def get_relevant_tools(self, job_title: str, job_desc: str):
        """
        Retrieving relevant jobs end extract common tools/frameworks mentioned in them
        """
        # retrieval
        formatted_for_retrieving = format_for_retriever(job_title = job_title, job_desc = job_desc)
        retrieved = retrieve(
            retriever_query = formatted_for_retrieving,
            embedding_model = CFG.EMBEDDING_MODEL,
            cross_encoder = CFG.RERANKER,
            collection = self.collection,
        )

        formatted = format_retrieved_docs(retrieved)

        # query llm to extract skills
        system_prompt = get_tools_prompt()
        tools = extract_tools(
            client = self.client,
            query = formatted,
            model_name = self.skills_extractor,
            system_prompt = system_prompt,
            temperature = 0.7
        )

        return tools
    # -------------------------------------------------------------------------------------
    def enhnace(self, job_info: dict, stream = False, debug = False, **kwargs) -> str:
        """
        Enhances an existing job description using an LLM

        Args:
            job_info: dictionary contains the job info (title - desc)
            use_rag: whether to use RAG or not
            stream: If True, returns a streaming response. Defaults to False.
            **kwargs: Additional parameters passed directly to the model API
                (e.g., temperature, max_tokens).
        """
        # extract data
        job_title = job_info["title"],
        job_desc = job_info["description"]

        # detect existing skills
        has_tools = self.detect_tools(job_desc = job_desc)

        if not has_tools:
            relevant_tools = self.get_relevant_tools(job_title = job_title, job_desc = job_desc)
            relevant_tools = list(set(ast.literal_eval(relevant_tools)))[:20]
            # filter the tools accepted by the client
            formatted = format_for_enhancement(job_disc = job_desc, tools = relevant_tools)
            system_prompt = get_enhancement_prompt(use_rag = True)
            response = enhance_old_job(
                client = self.client,
                query = formatted,
                model_name = self.enhancement_model,
                system_prompt = system_prompt,
                stream = stream,
                **kwargs
            )
        
        else:
            relevant_tools = None
            formatted = format_for_enhancement(job_disc = job_desc)
            system_prompt = get_enhancement_prompt(use_rag = False)
            response = enhance_old_job(
                client = self.client,
                query = formatted,
                model_name = self.enhancement_model,
                system_prompt = system_prompt,
                stream = stream,
                **kwargs
            )
        
        if debug:
            return {
                "has_tools" : has_tools,
                "relevant_tools" : relevant_tools,
                "response" : response
            }
        
        return response
    # -------------------------------------------------------------------------------------
    def close_db(self):
        self.weaviate_client.close()