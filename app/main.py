import os
import io
import re
import logging
import pdfplumber
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from celery.result import AsyncResult
from docx import Document as DocxDocument
from app.tasks import analyze_script, celery_app
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
app = FastAPI(title="Parents Guide")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def clean_text(text: str) -> str:
    """Remove non-printable characters and extra spaces."""
    if not text:
        return ""
    text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


async def extract_text_from_upload(file: UploadFile) -> str:
    ext = os.path.splitext(file.filename)[1].lower()

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="The file is empty.")

    try:
        if ext == ".pdf":
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                text = "\n".join(page.extract_text() or "" for page in pdf.pages)
        elif ext == ".docx":
            doc = DocxDocument(io.BytesIO(file_bytes))
            text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        else:
            raise HTTPException(status_code=400, detail="Only PDF and DOCX are supported.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading file: {str(e)}")

    if not text.strip():
        raise HTTPException(status_code=400, detail="The file does not contain text.")

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
        raise HTTPException(status_code=404, detail="Task not found")

    if ar.state in ["PENDING", "PROGRESS"]:
        return {"status": ar.state.lower(), "meta": ar.info or {}}

    if ar.successful():
        result = ar.result
        return {"status": "success", "result": result}

    if ar.failed():
        return {"status": "failed", "error": str(ar.result)}

    return {"status": ar.state, "info": ar.info or {}}


FONT_PATH = "/usr/share/fonts/Adwaita/AdwaitaMono-BoldItalic.ttf"  # или путь к другому TTF
pdfmetrics.registerFont(TTFont("DejaVu", FONT_PATH))

# Создаем стили с этим шрифтом
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='RussianNormal', parent=styles['Normal'], fontName="DejaVu"))
styles.add(ParagraphStyle(name='RussianHeading', parent=styles['Heading3'], fontName="DejaVu"))
styles.add(ParagraphStyle(name='RussianTitle', parent=styles['Title'], fontName="DejaVu"))


@app.get("/download_pdf/{task_id}")
def download_pdf(task_id: str):
    # Получаем результат задачи Celery
    ar = AsyncResult(task_id, app=celery_app)
    if not ar or not ar.successful():
        raise HTTPException(status_code=404, detail="Result not found")

    result = ar.result
    pdf_path = f"/tmp/{task_id}.pdf"

    # Создаем PDF
    doc = SimpleDocTemplate(pdf_path, pagesize=A4)
    story = []

    # Заголовок
    story.append(Paragraph("Parents Guide — Анализ сценария", styles["RussianTitle"]))
    story.append(Spacer(1, 12))

    # AgeCategory
    age = result.get("AgeCategory", "Не указано")
    story.append(Paragraph(f"<b>Возрастная категория:</b> {age}", styles["RussianNormal"]))
    story.append(Spacer(1, 12))

    # ParentsGuide
    pg = result.get("ParentsGuide", {})
    for category, data in pg.items():
        sev = data.get("Severity", "Нет")
        reason = data.get("Reason", "Нет данных")

        story.append(Paragraph(f"<b>{category}</b> — {sev}", styles["RussianHeading"]))
        story.append(Paragraph(reason.replace("\n", "<br/>"), styles["RussianNormal"]))
        story.append(Spacer(1, 12))

    # Генерация PDF
    doc.build(story)

    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename="parents_guide.pdf",
    )