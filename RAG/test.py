import os
import time
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_core.prompts import PromptTemplate

# =========================
# 1. Load API Key
# =========================
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("❌ GOOGLE_API_KEY not found. Check your .env file.")

# =========================
# 2. Load Documents
# =========================
KNOWLEDGE_BASE_DIR = "./RAG Knowledge Base"

print("\n--- Step 1: Loading Documents ---")
loader = DirectoryLoader(
    KNOWLEDGE_BASE_DIR,
    glob="**/*.txt",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"}
)
docs = loader.load()
print(f"✅ Loaded {len(docs)} documents.")

# =========================
# 3. Split Documents
# =========================
print("\n--- Step 2: Splitting Documents ---")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=3000,
    chunk_overlap=300
)
splits = text_splitter.split_documents(docs)
print(f"✅ Created {len(splits)} chunks.")

# =========================
# 4. Create Vector DB (with rate limit handling)
# =========================
print("\n--- Step 3: Creating Vector Database ---")
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=api_key
)

BATCH_SIZE = 20
vector_db = None

for i in range(0, len(splits), BATCH_SIZE):
    batch = splits[i:i + BATCH_SIZE]
    total_batches = (len(splits) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"  Batch {i//BATCH_SIZE + 1}/{total_batches} — {len(batch)} chunks...")

    success = False
    while not success:
        try:
            if vector_db is None:
                vector_db = Chroma.from_documents(
                    documents=batch,
                    embedding=embeddings,
                    persist_directory="./chroma_db"
                )
            else:
                vector_db.add_documents(batch)
            success = True
            if i + BATCH_SIZE < len(splits):
                print(f"  ✅ Batch done. Waiting 65s to respect rate limit...")
                time.sleep(65)

        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                print(f"  ⚠️ Rate limit hit. Waiting 70s before retry...")
                time.sleep(70)
            else:
                raise e

retriever = vector_db.as_retriever(search_kwargs={"k": 3})
print("✅ Vector DB ready.")

# =========================
# 5. Setup LLM (Gemini)
# =========================
print("\n--- Step 4: Loading Gemini ---")
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2,
    google_api_key=api_key
)

# =========================
# 6. Create RAG Chain
# =========================
print("\n--- Step 5: Creating RAG Chain ---")

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a professional Technical Recruiter.
Based ONLY on the provided documents below, extract skills, tools, and requirements for the role.
If information is missing, say that clearly.

Context:
{context}

Question: {question}

Format output as:
- Role Overview
- Core Skills
- Tools & Technologies
"""
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=False,
    chain_type_kwargs={"prompt": prompt}
)
print("✅ RAG System Ready.")

# =========================
# 7. Ask Function
# =========================
def ask_rag_system(user_input):
    print(f"\n🔍 Searching for: {user_input}")
    result = qa_chain.invoke({"query": user_input})
    print("\n--- 🤖 System Response ---")
    print(result["result"])

# =========================
# 8. Run Test
# =========================
if __name__ == "__main__":
    ask_rag_system(
        "What are the core technical skills and tools for Block chain?"
    )