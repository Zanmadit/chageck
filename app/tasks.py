from celery import Celery
from app.pipeline import run_analysis
from app.config import settings

celery_app = Celery(
    "tasks",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL, 
)

# Serialization options (JSON recommended)
celery_app.conf.update(
    task_serializer='json',
    result_serializer='json',
    accept_content=['json']
)

@celery_app.task(bind=True)
def analyze_script(self, script_text: str):
    return run_analysis(script_text)
