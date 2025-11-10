from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
import json

# Инициализация LLM
llm = ChatOllama(model="gemma3:4b")

CLASSIFY_PROMPT = """
You are a parental content classifier.
Analyze the {script_fragment} and output JSON like this:

{{
  "AgeCategory": ["16+", "18+"...],
  "ParentsGuide": {{
    "Sex & Nudity": ["None", "Mid", "Moderate", "Severe"],
    "Violence & Gore": ["None", "Mid", "Moderate", "Severe"],
    "Profanity": ["None", "Mid", "Moderate", "Severe"],
    "Alcohol, Drugs & Smoking": ["None", "Mid", "Moderate", "Severe"],
    "Frightening & Intense Scenes": ["None", "Mid", "Moderate", "Severe"]
  }},
  "Summary": "Brief reasoning."
}}
"""

# Создаем prompt
prompt = PromptTemplate(
    template=CLASSIFY_PROMPT,
    input_variables=["script_fragment"]
)

# Конвейер LCEL
chain = prompt | llm

def split_into_scenes(script_text: str):
    scenes = []
    current_scene = []
    for line in script_text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.upper().startswith(("СЦЕНА", "INT.", "EXT.")):
            if current_scene:
                scenes.append("\n".join(current_scene))
                current_scene = []
        current_scene.append(line)
    if current_scene:
        scenes.append("\n".join(current_scene))
    return scenes

def safe_parse(result):
    if not isinstance(result, str):
        result = str(result)
    try:
        return json.loads(result)
    except json.JSONDecodeError:
        try:
            return eval(result)
        except Exception:
            return {"AgeCategory": ["Unknown"], "ParentsGuide": {}}
        
def analyze_scenes(script_text: str):
    scenes = split_into_scenes(script_text)
    results = []

    for scene in scenes:
        output = chain.invoke({"script_fragment": scene})
        results.append(output.content)  # ✅ добавили .content

    severity_order = ["None", "Mid", "Moderate", "Severe"]
    parsed = [safe_parse(r) for r in results]

    final_result = {
        "AgeCategory": max([p.get("AgeCategory", ["Unknown"])[0] for p in parsed]),
        "ParentsGuide": {},
        "Summary": "Aggregated scene analysis."
    }

    categories = ["Sex & Nudity", "Violence & Gore", "Profanity",
                  "Alcohol, Drugs & Smoking", "Frightening & Intense Scenes"]

    for cat in categories:
        values = []
        for p in parsed:
            if "ParentsGuide" in p and cat in p["ParentsGuide"]:
                values.append(p["ParentsGuide"][cat][0])
        if values:
            final_result["ParentsGuide"][cat] = max(values, key=lambda x: severity_order.index(x))
        else:
            final_result["ParentsGuide"][cat] = "None"

    return final_result
