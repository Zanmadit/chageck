from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from langchain_core.documents import Document
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from app.jsonformer import OllamaJsonformer
from app.config import settings
from pathlib import Path
import json

PROMPT_PATH = Path("prompts/classify_prompt.txt")

CLASSIFY_PROMPT = PROMPT_PATH.read_text(encoding="utf-8")

llm = ChatOllama(model=settings.MODEL_NAME)

embeddings = OllamaEmbeddings(model=settings.EMBED_MODEL)

law_vector_store = QdrantVectorStore.from_existing_collection(
    url=settings.QDRANT_URL,
    collection_name=settings.LAW_COLLECTION,
    embedding=embeddings
)

law_retriever = law_vector_store.as_retriever(search_kwargs={"k": 5})

output_parser = JsonOutputParser()

client = QdrantClient(url=settings.QDRANT_URL)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=settings.CHUNK_SIZE, 
    chunk_overlap=settings.CHUNK_OVERLAP,
    length_function=len
)

prompt = ChatPromptTemplate.from_template(CLASSIFY_PROMPT)

# JSON schema for OllamaJsonFormer
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


def create_vector_store(script_text: str):
    chunks = splitter.split_text(script_text)
    docs = [Document(page_content=chunk) for chunk in chunks]

    client = QdrantClient(url=settings.QDRANT_URL)

    # Delete collection, if exists
    if client.collection_exists(settings.QDRANT_COLLECTION):
        client.delete_collection(settings.QDRANT_COLLECTION)

    vectorstore = QdrantVectorStore.from_documents(
        docs,
        embeddings,
        url=settings.QDRANT_URL,
        collection_name=settings.QDRANT_COLLECTION
    )

    return vectorstore.as_retriever(search_kwargs={"k": 5})


def determine_age_category(parents_guide: dict) -> str:
    severities = []
    for v in parents_guide.values():
        if isinstance(v, dict):
            s = v.get("Severity", "")
        else:
            s = str(v)
        s = (s or "").strip().lower()
        severities.append(s)

    strong_count = sum("сильный" in s for s in severities)
    medium_count = sum("средний" in s for s in severities)
    weak_count = sum("слабый" in s for s in severities)
    none_count = sum("нет" in s or s == "" for s in severities)
    total = len(severities)

    # Rules used
    if strong_count >= 2:
        return "18+"
    if medium_count >= 2 and strong_count == 0: 
        return "16+"
    if weak_count == total:
        return "12+"
    if weak_count >= 2 and (strong_count == 0 and medium_count == 0):
        return "6+"
    if none_count == total:
        return "0+"


def run_analysis(script_text: str):
    query = "Classify the parental content rating of this script."
    retriever = create_vector_store(script_text)

    def combine_docs(docs):
        texts = [d.page_content for d in docs if d.page_content and d.page_content.strip()]
        return "\n\n".join(texts)


    law_docs = law_vector_store.similarity_search(
        "возрастная классификация, сцены насилия, эротика, нецензурная лексика, наркотики, пугающие сцены",
        k=5
    )
    law_context = "\n".join([doc.page_content for doc in law_docs])

    chain = (
        retriever
        | combine_docs
        | (lambda script_text: {"script": script_text, "context": law_context})
        | prompt
        | llm
        | output_parser
    )

    try:
        result = chain.invoke(query)

        if isinstance(result, str):
            try:
                data = json.loads(result)
            except json.JSONDecodeError:
                data = {"AgeCategory": "Unknown", "ParentsGuide": {}, "Summary": "Invalid JSON output"}
        elif isinstance(result, dict):
            data = result
        else:
            data = {"AgeCategory": "Unknown", "ParentsGuide": {}, "Summary": "Unexpected output type"}

        parents_guide = data.get("ParentsGuide", {})
        computed_category = determine_age_category(parents_guide)

        if "AgeCategory" not in data or data["AgeCategory"] not in ["0+", "6+", "12+", "16+", "18+"]:
            data["AgeCategory"] = computed_category
        else:
            order = {"0+": 0, "6+": 1, "12+": 2, "16+": 3, "18+": 4}
            if order.get(computed_category, 0) > order.get(data["AgeCategory"], 0):
                data["AgeCategory"] = computed_category

        return data


    except (OutputParserException, json.JSONDecodeError):
        json_prompt = f"Analyze the following movie script and output classification as JSON:\n\n{script_text}"
        jsonformer = OllamaJsonformer(llm, json_schema, json_prompt)
        return jsonformer()

    except Exception as e:
        return {
            "AgeCategory": "Unknown",
            "ParentsGuide": {},
            "Summary": f"Error: {str(e)}"
        }
