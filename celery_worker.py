from celery import Celery

# Redis를 메시지 브로커(주문서 꽂이)로 사용합니다.
# 로컬(같은 컴퓨터)에서 Redis를 기본 포트로 켰을 때의 주소입니다.
celery_app = Celery(
    "video_tasks",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/0"
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="Asia/Seoul",
    enable_utc=True,
)