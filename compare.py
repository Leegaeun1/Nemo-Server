import json, os

def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    return inter / (area1 + area2 - inter + 1e-6)


video_name = "video_01"  # 이것만 바꾸면 됨


labels_dir = f"data/{video_name}/labels_yolo"
labelme_dir = f"data/{video_name}/labels_labelme"
frames_dir = f"data/{video_name}/frames"

results_summary = []

for json_file in sorted(os.listdir(labelme_dir)):
    if not json_file.endswith(".json"):
        continue

    # LabelMe 수정본
    with open(os.path.join(labelme_dir, json_file)) as f:
        lm = json.load(f)
    gt_boxes = [[s["points"][0][0], s["points"][0][1],
                  s["points"][1][0], s["points"][1][1]] for s in lm["shapes"]]

    # YOLO 예측본
    txt_file = json_file.replace(".json", ".txt")
    txt_path = os.path.join(labels_dir, txt_file)
    pred_boxes = []
    if os.path.exists(txt_path):
        img_w, img_h = lm["imageWidth"], lm["imageHeight"]
        with open(txt_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5: continue
                _, xc, yc, w, h = map(float, parts[:5])
                pred_boxes.append([
                    (xc-w/2)*img_w, (yc-h/2)*img_h,
                    (xc+w/2)*img_w, (yc+h/2)*img_h
                ])

    # IoU 계산
    ious = []
    for gt in gt_boxes:
        if pred_boxes:
            best = max(iou(gt, p) for p in pred_boxes)
            ious.append(best)

    avg_iou = sum(ious)/len(ious) if ious else 0
    results_summary.append({
        "file": json_file,
        "gt_count": len(gt_boxes),
        "pred_count": len(pred_boxes),
        "avg_iou": round(avg_iou, 3)
    })

for r in results_summary:
    print(r)