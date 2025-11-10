from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from transformers import AutoTokenizer, AutoModelForCausalLM
from jsonformer import Jsonformer
import json

MODEL_NAME = "NousResearch/Hermes-3-Llama-3.2-3B"
QDRANT_URL = "http://localhost:6333"
QDRANT_COLLECTION = "scripts_rag"

# --- Основные объекты ---
llm = ChatOllama(model=MODEL_NAME)
embeddings = OllamaEmbeddings(model="all-minilm")
output_parser = JsonOutputParser()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True)

client = QdrantClient(url=QDRANT_URL)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    length_function=len
)

# --- Создание векторного хранилища ---
def create_vector_store(script_text: str):
    chunks = splitter.split_text(script_text)
    docs = [Document(page_content=chunk) for chunk in chunks]

    vectorstore = QdrantVectorStore.from_documents(
        docs,
        embeddings,
        url=QDRANT_URL,
        collection_name=QDRANT_COLLECTION
    )
    return vectorstore.as_retriever(search_kwargs={"k": 3})

# --- Промпт для классификации ---
CLASSIFY_PROMPT = """
RETURN ONLY VALID JSON.
You are a parental content classifier.
Given the following movie script fragments, analyze them and output JSON like this:

{{
  "AgeCategory": ["16+", "18+"],
  "ParentsGuide": {{
    "Sex & Nudity": ["None", "Mid", "Moderate", "Severe"],
    "Violence & Gore": ["None", "Mid", "Moderate", "Severe"],
    "Profanity": ["None", "Mid", "Moderate", "Severe"],
    "Alcohol, Drugs & Smoking": ["None", "Mid", "Moderate", "Severe"],
    "Frightening & Intense Scenes": ["None", "Mid", "Moderate", "Severe"]
  }},
  "Summary": "Brief reasoning."
}}

Script context:
{context}

Now classify it:
"""

prompt = ChatPromptTemplate.from_template(CLASSIFY_PROMPT)

# --- JSON схема для Jsonformer ---
json_schema = {
    "type": "object",
    "properties": {
        "AgeCategory": {"type": "string"},
        "ParentsGuide": {
            "type": "object",
            "properties": {
                "Sex & Nudity": {"type": "string"},
                "Violence & Gore": {"type": "string"},
                "Profanity": {"type": "string"},
                "Alcohol, Drugs & Smoking": {"type": "string"},
                "Frightening & Intense Scenes": {"type": "string"}
            }
        },
        "Summary": {"type": "string"}
    },
    "required": ["AgeCategory", "ParentsGuide", "Summary"]
}

# --- Основная функция анализа ---
def run_analysis(script_text: str):
    retriever = create_vector_store(script_text)
    chain = retriever | prompt | llm | output_parser

    query = "Classify the parental content rating of this script."

    try:
        result = chain.invoke(query)
        data = json.loads(result["result"]) if isinstance(result, dict) else json.loads(result)
        return data

    except (OutputParserException, json.JSONDecodeError):
        # Fallback на Jsonformer, если LLM не вернул корректный JSON
        json_prompt = f"Analyze the following movie script and output classification as JSON:\n\n{script_text}"
        jsonformer = Jsonformer(model, json_prompt,)
        data = jsonformer()
        return data

    except Exception as e:
        # На всякий случай общий fallback
        return {
            "AgeCategory": "Unknown",
            "ParentsGuide": {},
            "Summary": f"Error: {str(e)}"
        }
