from fastapi import FastAPI, UploadFile, HTTPException
from celery.result import AsyncResult
import chardet
from app.tasks import analyze_script, celery_app
import logging

logger = logging.getLogger(__name__)
app = FastAPI(title="Parental Content Classifier")

@app.post("/upload")
async def upload_script(file: UploadFile):
    try:
        raw_bytes = await file.read()
        detected = chardet.detect(raw_bytes)
        encoding = detected['encoding'] or 'utf-8'
        text = raw_bytes.decode(encoding, errors='replace')
    except Exception as e:
        logger.exception("Failed to read uploaded file")
        raise HTTPException(status_code=400, detail="Cannot read uploaded file")

    task = analyze_script.delay(text)
    return {"task_id": task.id}

@app.get("/result/{task_id}")
def get_result(task_id: str):
    try:
        ar = AsyncResult(task_id, app=celery_app)
        if not ar:
            raise HTTPException(status_code=404, detail="Task not found")

        if ar.state == 'PENDING':
            return {"status": "pending"}
        if ar.state == 'PROGRESS':
            # если используешь update_state, можно вернуть meta
            return {"status": "in_progress", "meta": ar.info or {}}

        if ar.successful():
            result = ar.result  # безопасно: backend возвращает реальный результат
            # дополнительно — проверим тип и вернём структурированный ответ
            if isinstance(result, dict):
                return {"status": "success", "result": result}
            else:
                # возможен случай: результат — строка JSON
                try:
                    import json
                    parsed = json.loads(result)
                    return {"status": "success", "result": parsed}
                except Exception:
                    return {"status": "success", "result": {"raw": str(result)}}

        if ar.failed():
            # ar.result может быть Exception
            exc = ar.result
            logger.error("Task failed: %s", exc)
            return {"status": "failed", "error": str(exc)}

        # для других состояний
        return {"status": ar.state, "info": ar.info or {}}
    except Exception as e:
        logger.exception("Error fetching task result")
        raise HTTPException(status_code=500, detail="Internal server error")
