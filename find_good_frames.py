# find_good_frames.py
import json, sys, os

VIDEO = sys.argv[1] if len(sys.argv) > 1 else "video_03"
RES = "results_reset/yolov12s-face"

def iou(a, b):
    x1=max(a[0],b[0]); y1=max(a[1],b[1]); x2=min(a[2],b[2]); y2=min(a[3],b[3])
    i=max(0,x2-x1)*max(0,y2-y1)
    return i/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-i+1e-6)

new = json.load(open(f"{RES}/{VIDEO}_coords_new.json"))
gt_dir = f"data/{VIDEO}/labels_labelme"
out = []
for jf in sorted(os.listdir(gt_dir)):
    if not jf.endswith(".json"): continue
    key = jf[:-5]
    if key not in new: continue
    lm = json.load(open(os.path.join(gt_dir, jf)))
    gt = [[min(s["points"][0][0], s["points"][1][0]), min(s["points"][0][1], s["points"][1][1]),
           max(s["points"][0][0], s["points"][1][0]), max(s["points"][0][1], s["points"][1][1])]
          for s in lm["shapes"]]
    e = new[key]
    fwd = e["fwd_raw"] + e["fwd_rec_all"]
    rev = e["rev_raw_all"]
    rescued = [g for g in gt
               if not any(iou(g, f) >= 0.3 for f in fwd)
               and any(iou(g, r) >= 0.3 for r in rev)]
    if rescued:
        out.append((len(rescued), len(gt), key))
out.sort(reverse=True)
print("정방향이 놓친 GT를 역방향이 복원한 프레임 (복원수, GT수, 프레임):")
for r in out[:15]:
    print(" ", r)