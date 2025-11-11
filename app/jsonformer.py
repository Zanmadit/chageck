from langchain_ollama import ChatOllama
import json

class OllamaJsonformer:
    def __init__(self, llm: ChatOllama, json_schema: dict, prompt: str):
        self.llm = llm
        self.json_schema = json_schema
        self.prompt = prompt

    def __call__(self):
        full_prompt = f"{self.prompt}\nOutput JSON schema:\n{json.dumps(self.json_schema)}\nReturn only valid JSON."
        try:
            resp = self.llm.invoke({"prompt": full_prompt})
            resp_text = getattr(resp, "content", str(resp))
            return json.loads(resp_text)
        except json.JSONDecodeError:
            return self.empty_result()
        except Exception as e:
            return {"error": str(e)}

    def empty_result(self):
        result = {}
        for key, val in self.json_schema.get("properties", {}).items():
            if val["type"] == "object":
                result[key] = {k: "Unknown" for k in val.get("properties", {})}
            elif val["type"] == "array":
                result[key] = []
            else:
                result[key] = "Unknown"
        return result