import cv2
for name in ["forward", "reverse", "merged"]:
    img = cv2.imread(f"fig_video_03_frame_0169_{name}.png")
    crop = img[300:1300, 150:2800]   # 확인하며 조정
    cv2.imwrite(f"crop_0169_{name}.png", crop)