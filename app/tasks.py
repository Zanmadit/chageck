from celery import Celery
from app.pipeline import run_analysis

celery_app = Celery(
    "tasks",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/0",  # <- ОБЯЗАТЕЛЬНО укажи backend
)

# Опции сериализации (рекомендуется JSON)
celery_app.conf.update(
    task_serializer='json',
    result_serializer='json',
    accept_content=['json']
)

@celery_app.task(bind=True)
def analyze_script(self, script_text: str):
    # Можно логировать прогресс: self.update_state(state='PROGRESS', meta={...})
    return run_analysis(script_text)
