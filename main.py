from fastapi import FastAPI, UploadFile, File, Form
import shutil
import os
import uuid
import glob
from celery_worker import celery_app
import tasks  # tasks.py를 불러와야 Celery가 작업을 인식합니다.
from fastapi.responses import FileResponse
from fastapi.responses import JSONResponse
import hashlib
from starlette.background import BackgroundTask

app = FastAPI()

FACES_DIR = "user_faces"  # 얼굴 이미지 저장 폴더
os.makedirs(FACES_DIR, exist_ok=True) 

# 영상이 저장될 폴더 만들기
os.makedirs("uploads", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

@app.post("/upload")
async def upload_video(
    video: UploadFile = File(...),
    device_token: str = Form(...),
    user_id: str = Form(...)  # 추가
):
    file_id = str(uuid.uuid4())
    input_path = f"uploads/{file_id}_{video.filename}"
    output_path = f"outputs/{file_id}_result.mp4"

    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)

    # device_token(알림용)과 user_id(폴더식별용) 둘 다 전달
    tasks.process_video_task.delay(input_path, output_path, device_token, user_id)

    return {"message": "접수 완료", "task_id": file_id}

@app.post("/upload_face")
async def upload_face(
    file: UploadFile = File(...),
    device_token: str = Form(...),
    user_id: str = Form(...)  # 추가
):
    user_dir = f"user_faces/{user_id}"  # 해시 불필요, user_id는 UUID라 안전
    os.makedirs(user_dir, exist_ok=True)

    existing = glob.glob(f"{user_dir}/person*.*")
    next_index = len(existing) + 1
    ext = os.path.splitext(file.filename)[-1]
    save_path = f"{user_dir}/person{next_index}{ext}"

    with open(save_path, "wb") as f:
        f.write(await file.read())

    return {"status": "ok", "path": save_path, "index": next_index}

@app.get("/download/{filename}")
async def download_video(filename: str):
    # output 파일 저장 경로 (tasks.py의 output_path와 맞춰야 함)
    file_path = f"outputs/{filename}"
    
    if not os.path.exists(file_path):
        return {"error": "파일이 없습니다."}
    
    response = FileResponse(
        path=file_path,
        media_type="video/mp4",
        filename=filename,
        background=BackgroundTask(lambda: os.remove(file_path) if os.path.exists(file_path) else None)
    )
    return response