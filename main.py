from fastapi import FastAPI, UploadFile, File, Form
import shutil
import os
import uuid
from celery_worker import celery_app
import tasks  # tasks.py를 불러와야 Celery가 작업을 인식합니다.

app = FastAPI()

# 영상이 저장될 폴더 만들기
os.makedirs("uploads", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

@app.post("/upload")
async def upload_video(
    video: UploadFile = File(...),
    device_token: str = Form(...)  # 플러터에서 보낼 파이어베이스 기기 토큰
):
    # 1. 고유한 파일 이름 생성 (여러 명이 동시에 올려도 섞이지 않게)
    file_id = str(uuid.uuid4())
    input_path = f"uploads/{file_id}_{video.filename}"
    output_path = f"outputs/{file_id}_result.mp4"

    # 2. 업로드된 파일을 서버 'uploads' 폴더에 저장
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)

    # 3. Celery 주방장에게 비식별화 작업 지시 (비동기)
    # .delay()를 쓰면 작업이 끝날 때까지 기다리지 않고 백그라운드로 던집니다.
    tasks.process_video_task.delay(input_path, output_path, device_token)

    # 4. 앱에는 즉시 '접수 완료'라고 응답 (앱 멈춤 방지)
    return {
        "message": "비디오가 성공적으로 접수되었습니다. 백그라운드에서 처리를 시작합니다.",
        "task_id": file_id
    }