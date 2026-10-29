# make_figure_panels.py
import cv2, json, sys
import numpy as np

VIDEO = sys.argv[1] if len(sys.argv) > 1 else "video_03"
RES   = "results_reset/yolov12s-face"
GREEN, RED = (60, 180, 60), (50, 50, 220)   # BGR

def iou(a, b):
    x1=max(a[0],b[0]); y1=max(a[1],b[1]); x2=min(a[2],b[2]); y2=min(a[3],b[3])
    i=max(0,x2-x1)*max(0,y2-y1)
    return i/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-i+1e-6)

def draw(img, boxes, color, dashed_boxes=(), th=3):
    out = img.copy()
    for x1,y1,x2,y2 in boxes:
        cv2.rectangle(out,(int(x1),int(y1)),(int(x2),int(y2)),color,th)
    for x1,y1,x2,y2 in dashed_boxes:           # 히트맵 복원 박스는 점선 느낌으로 구분
        x1,y1,x2,y2 = map(int,(x1,y1,x2,y2))
        for x in range(x1,x2,12):
            cv2.line(out,(x,y1),(min(x+6,x2),y1),color,th); cv2.line(out,(x,y2),(min(x+6,x2),y2),color,th)
        for y in range(y1,y2,12):
            cv2.line(out,(x1,y),(x1,min(y+6,y2)),color,th); cv2.line(out,(x2,y),(x2,min(y+6,y2)),color,th)
    return out

def label(img, text):
    cv2.rectangle(img,(0,0),(img.shape[1],44),(255,255,255),-1)
    cv2.putText(img,text,(12,32),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,0,0),2,cv2.LINE_AA)
    return img

new = json.load(open(f"{RES}/{VIDEO}_coords_new.json"))

# 후보: 정방향(YOLO+히트맵)엔 없고 역방향 YOLO 박스만 새로 잡은 프레임
cands = []
for k, e in new.items():
    fwd = e["fwd_raw"] + e["fwd_rec_all"]
    extra = [b for b in e["rev_raw_all"] if not any(iou(b, f) >= 0.3 for f in fwd)]
    if extra:
        cands.append((len(extra), k))
cands.sort(reverse=True)
print("후보 프레임(상위 10):", cands[:10])

key = sys.argv[2] if len(sys.argv) > 2 else cands[0][1]      # 직접 고르려면 frame_0123 형태로 인자 전달
e = new[key]
img = cv2.imread(f"data/{VIDEO}/frames/{key}.jpg")

fwd_panel = label(draw(img, e["fwd_raw"], GREEN, e["fwd_rec_all"]), "Forward (1 -> T)")
rev_panel = label(draw(img, e["rev_raw_all"], RED, e["rev_rec_all"]), "Reverse (T -> 1)")
merged    = label(draw(img, e["yolo"] + e["heatmap"], GREEN,
                       [], 3), "Merged")
# 병합 패널: 정방향 유래=초록, 역방향으로 추가된 박스=빨강
merged = draw(merged, e["reverse"] + e["reverse_heat"], RED)

cv2.imwrite(f"fig_{VIDEO}_{key}_forward.png", fwd_panel)
cv2.imwrite(f"fig_{VIDEO}_{key}_reverse.png", rev_panel)
cv2.imwrite(f"fig_{VIDEO}_{key}_merged.png",  merged)
cv2.imwrite(f"fig_{VIDEO}_{key}_all.png", np.hstack([fwd_panel, rev_panel, merged]))
print("저장 완료:", key)