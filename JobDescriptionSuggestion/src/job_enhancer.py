# ---------------------------------------------------------------------------
# Contains the required class to
# - Initiate the enhancer
# - enhancing the given job desc
# - detect the skills
# 
# Eng. Sara
# ---------------------------------------------------------------------------



from connect_to_groq import get_groq_client, enhance_old_job, has_skills
from vector_database import get_weaviate_client, load_collection, retrieve
from prompts import get_enhancement_prompt, get_detection_prompt

class Enhancer:
    """
    General class for Job Description Enhancement / Suggestion
    """
    def __init__(self, enhancement_model: str, detection_model: str, model_provider = 'groq'):
        self.enhancement_model = enhancement_model
        self.detection_model = detection_model
        
        
        self.groq_client = get_groq_client()
        self.weaviate_client = get_weaviate_client()
        
        # load db
        self.collection = load_collection(self.weaviate_client, config.COLLECTION_NAME)

    def detect_skills(self, query: str) -> bool:
        """
        يستخدم دالة حنين (has_skills) ليعرف إذا كان النص يحتوي على مهارات
        """
        system_prompt = get_detection_prompt()
        response = has_skills(
            client=self.groq_client,
            query=query,
            model_name=self.detection_model,
            system_prompt=system_prompt,
            temperature=0 
        )
        
        return "yes" in response.lower()

    def enhance_old_desc(self, query: str, use_rag: bool, stream = False, **kwargs) -> str:
        """
        الدالة الأساسية لتحسين الوصف الوظيفي
        """
        retrieved_docs_text = []
        
        if use_rag:
            results = retrieve(
                retriever_query=query,
                embedding_model=config.EMBEDDING_MODEL,
                cross_encoder=config.RERANKER,
                collection=self.collection,
                n_to_return=5 
            )
            retrieved_docs_text = [doc[0].properties.get("job_document") for doc in results]

        system_prompt = get_enhancement_prompt(use_rag=use_rag, retrieved_documents=retrieved_docs_text)

        response = enhance_old_job(
            client=self.groq_client,
            query=query,
            model_name=self.enhancement_model,
            system_prompt=system_prompt,
            use_rag=use_rag,
            retrieved_documents=retrieved_docs_text,
            stream=stream,
            **kwargs
        )

        return response