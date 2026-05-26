from fastapi import FastAPI, UploadFile, File, Form
import shutil
import os
import uuid
import glob
from celery_worker import celery_app
import tasks_add_backTracking_V2 as tasks  # tasks.py를 불러와야 Celery가 작업을 인식합니다.
from fastapi.responses import FileResponse
from fastapi.responses import JSONResponse
import hashlib
from starlette.background import BackgroundTask
import redis
from typing import Dict
from fastapi import Body
app = FastAPI()
redis_client = redis.Redis(host='localhost', port=6379, db=0)
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
    result = tasks.process_video_task.delay(input_path, output_path, device_token, user_id)

    queue_length = redis_client.llen('celery')  # 대기 중인 작업 수
    active_count = 1 if redis_client.llen('celery') >= 0 else 0  # 현재 처리 중
    queue_position = queue_length  # 방금 넣은 것 포함

    return {"message": "접수 완료", "task_id": file_id, "queue_position": queue_position}

@app.delete("/delete_face")
async def delete_face(body: Dict = Body(...)):
    user_id = body.get('user_id')
    filename = body.get('filename')
    file_path = f"user_faces/{user_id}/{filename}"
    
    print(f"삭제 요청: {file_path}")  # ✅ 어떤 경로로 오는지 확인
    print(f"파일 존재 여부: {os.path.exists(file_path)}")
    
    if os.path.exists(file_path):
        os.remove(file_path)
        return {"status": "ok"}
    return {"status": "not found"}

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

@app.get("/file-info/{filename}")
async def get_file_info(filename: str):
    file_path = f"outputs/{filename}"
    if not os.path.exists(file_path):
        return JSONResponse(status_code=404, content={"error": "파일이 없습니다."})

    size_bytes = os.path.getsize(file_path)
    size_mb = size_bytes / (1024 * 1024)
    size_str = f"{size_mb:.1f}MB"

    thumbnail_filename = filename.replace(".mp4", ".jpg")
    thumbnail_path = f"outputs/{thumbnail_filename}"
    has_thumbnail = os.path.exists(thumbnail_path)

    return {
        "name": filename,
        "size": size_str,
        "thumbnail_url": f"/thumbnail/{thumbnail_filename}" if has_thumbnail else None,
    }


@app.get("/thumbnail/{filename}")
async def get_thumbnail(filename: str):
    thumbnail_path = f"outputs/{filename}"
    if not os.path.exists(thumbnail_path):
        return JSONResponse(status_code=404, content={"error": "썸네일이 없습니다."})
    return FileResponse(path=thumbnail_path, media_type="image/jpeg")


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