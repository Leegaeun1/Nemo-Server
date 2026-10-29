# check_frame.py
import cv2, json, sys

VIDEO = sys.argv[1]
KEY   = sys.argv[2]
RES   = "results_reset/yolov12s-face"

def iou(a, b):
    x1=max(a[0],b[0]); y1=max(a[1],b[1]); x2=min(a[2],b[2]); y2=min(a[3],b[3])
    i=max(0,x2-x1)*max(0,y2-y1)
    return i/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-i+1e-6)

e = json.load(open(f"{RES}/{VIDEO}_coords_new.json"))[KEY]
lm = json.load(open(f"data/{VIDEO}/labels_labelme/{KEY}.json"))
gt = [[min(s["points"][0][0], s["points"][1][0]), min(s["points"][0][1], s["points"][1][1]),
       max(s["points"][0][0], s["points"][1][0]), max(s["points"][0][1], s["points"][1][1])]
      for s in lm["shapes"]]

fwd = e["fwd_raw"] + e["fwd_rec_all"]
extra = [b for b in e["rev_raw_all"] if not any(iou(b, f) >= 0.3 for f in fwd)]

print(f"GT {len(gt)}개 / 정방향 박스 {len(fwd)}개 / 역방향이 새로 잡은 박스 {len(extra)}개")
for b in extra:
    best = max((iou(b, g) for g in gt), default=0)
    print(f"  역방향 추가 박스 {[int(v) for v in b]}  GT와 최대 IoU = {best:.2f}  ->",
          "실제 얼굴 (OK)" if best >= 0.3 else "오탐 의심")
for g in gt:
    best = max((iou(g, f) for f in fwd), default=0)
    print(f"  GT {[int(v) for v in g]}  정방향 최대 IoU = {best:.2f}  ->",
          "정방향이 잡음" if best >= 0.3 else "정방향이 놓침")

img = cv2.imread(f"data/{VIDEO}/frames/{KEY}.jpg")
for b in gt:    cv2.rectangle(img, tuple(map(int, b[:2])), tuple(map(int, b[2:])), (0, 255, 255), 2)  # GT 노랑
for b in fwd:   cv2.rectangle(img, tuple(map(int, b[:2])), tuple(map(int, b[2:])), (60, 180, 60), 2) # 정방향 초록
for b in extra: cv2.rectangle(img, tuple(map(int, b[:2])), tuple(map(int, b[2:])), (50, 50, 220), 3) # 역방향 추가 빨강
cv2.imwrite(f"check_{VIDEO}_{KEY}.png", img)
print(f"저장: check_{VIDEO}_{KEY}.png  (노랑=GT, 초록=정방향, 빨강=역방향이 추가)")