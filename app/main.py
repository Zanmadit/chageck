import os
import io
import re
import logging
import pdfplumber
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from celery.result import AsyncResult
from docx import Document as DocxDocument
from app.tasks import analyze_script, celery_app


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
app = FastAPI(title="PDF/DOCX Text Analyzer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def clean_text(text: str) -> str:
    """Удаляет непечатные символы и лишние пробелы."""
    if not text:
        return ""
    text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


async def extract_text_from_upload(file: UploadFile) -> str:
    ext = os.path.splitext(file.filename)[1].lower()

    # читаем как байты
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Файл пуст")

    try:
        if ext == ".pdf":
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                text = "\n".join(page.extract_text() or "" for page in pdf.pages)
        elif ext == ".docx":
            doc = DocxDocument(io.BytesIO(file_bytes))
            text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        else:
            raise HTTPException(status_code=400, detail="Поддерживаются только PDF и DOCX")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ошибка при чтении файла: {str(e)}")

    if not text.strip():
        raise HTTPException(status_code=400, detail="Файл не содержит текста")

    return clean_text(text)


@app.post("/upload")
async def upload_script(file: UploadFile):
    text = await extract_text_from_upload(file)
    task = analyze_script.delay(text)
    logger.info(f"Task created {task.id}")
    return {"task_id": task.id}


@app.get("/result/{task_id}")
def get_result(task_id: str):
    ar = AsyncResult(task_id, app=celery_app)
    if not ar:
        raise HTTPException(status_code=404, detail="Задача не найдена")

    if ar.state in ["PENDING", "PROGRESS"]:
        return {"status": ar.state.lower(), "meta": ar.info or {}}

    if ar.successful():
        result = ar.result
        return {"status": "success", "result": result}

    if ar.failed():
        return {"status": "failed", "error": str(ar.result)}

    return {"status": ar.state, "info": ar.info or {}}
