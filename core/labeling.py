import sys
import os
import glob
from PyQt6.QtWidgets import *
from PyQt6.QtGui import *
from PyQt6.QtCore import *

class YoloReviewer(QMainWindow):
    def __init__(self, root_path='dataset_extreme_occ'):
        super().__init__()
        self.root_path = root_path
        self.image_paths = glob.glob(os.path.join(self.root_path, 'images', '**', '*.jpg'), recursive=True)
        self.current_idx = 0
        
        # 드래그 관련 상태 변수
        self.begin = QPoint()
        self.end = QPoint()
        self.is_drawing = False
        self.current_pixmap = None
        
        self.initUI()
        if self.image_paths:
            self.load_image()

    def initUI(self):
        self.setWindowTitle('가은님의 스마트 라벨러 v2.0')
        self.setGeometry(100, 100, 1200, 900)
        
        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setCentralWidget(self.label)
        
        self.statusBar().showMessage("마우스 드래그: 박스 추가 | Space: 다음 | Delete: 삭제 | Left: 이전")

    def load_image(self):
        img_path = self.image_paths[self.current_idx]
        self.current_pixmap = QPixmap(img_path)
        self.update_display()

    def update_display(self):
        if not self.current_pixmap: return
        
        # 1. 원본 이미지 복사
        display_pixmap = self.current_pixmap.copy()
        painter = QPainter(display_pixmap)
        
        # 2. 기존 라벨 그리기 (초록색)
        txt_path = self.image_paths[self.current_idx].replace(os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep).replace('.jpg', '.txt')
        if os.path.exists(txt_path):
            painter.setPen(QPen(Qt.GlobalColor.green, 4))
            with open(txt_path, 'r') as f:
                for line in f:
                    parts = line.split()
                    if len(parts) < 5: continue
                    cls, cx, cy, w, h = map(float, parts[:5])
                    pw, ph = display_pixmap.width(), display_pixmap.height()
                    painter.drawRect(int((cx-w/2)*pw), int((cy-h/2)*ph), int(w*pw), int(h*ph))

        # 3. 현재 드래그 중인 박스 그리기 (빨간색 점선)
        if self.is_drawing:
            painter.setPen(QPen(Qt.GlobalColor.red, 3, Qt.PenStyle.DashLine))
            painter.drawRect(QRect(self.begin, self.end))
        
        painter.end()
        
        # 4. 레이블 크기에 맞게 조절하여 표시
        scaled = display_pixmap.scaled(self.label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.label.setPixmap(scaled)
        self.setWindowTitle(f"[{self.current_idx+1}/{len(self.image_paths)}] {os.path.basename(self.image_paths[self.current_idx])}")

    # --- 마우스 이벤트 ---
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            # QLabel 내의 실제 이미지 영역과 마우스 위치 계산 (이 부분은 단순화를 위해 좌표 직접 매핑)
            self.begin = self.map_to_pixmap(event.pos())
            self.end = self.begin
            self.is_drawing = True

    def mouseMoveEvent(self, event):
        if self.is_drawing:
            self.end = self.map_to_pixmap(event.pos())
            self.update_display()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.is_drawing:
            self.end = self.map_to_pixmap(event.pos())
            self.is_drawing = False
            self.save_new_box()
            self.load_image() # 다시 로드해서 그리기

    def map_to_pixmap(self, pos):
        # 화면 좌표를 원본 이미지 좌표로 변환하는 정교한 로직
        lbl_w, lbl_h = self.label.width(), self.label.height()
        pix_w, pix_h = self.current_pixmap.width(), self.current_pixmap.height()
        
        # 비율 계산
        scale = min(lbl_w/pix_w, lbl_h/pix_h)
        offset_x = (lbl_w - pix_w * scale) / 2
        offset_y = (lbl_h - pix_h * scale) / 2
        
        # QLabel 상대 좌표로 변환 후 스케일 역산
        rel_pos = self.label.mapFromParent(pos)
        actual_x = (rel_pos.x() - offset_x) / scale
        actual_y = (rel_pos.y() - offset_y) / scale
        
        return QPoint(int(actual_x), int(actual_y))

    def save_new_box(self):
        # YOLO 포맷으로 저장 (정규화)
        pw, ph = self.current_pixmap.width(), self.current_pixmap.height()
        
        x1, y1 = self.begin.x(), self.begin.y()
        x2, y2 = self.end.x(), self.end.y()
        
        # 박스 정규화 (0~1)
        cx = ((x1 + x2) / 2) / pw
        cy = ((y1 + y2) / 2) / ph
        w = abs(x2 - x1) / pw
        h = abs(y2 - y1) / ph
        
        txt_path = self.image_paths[self.current_idx].replace(os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep).replace('.jpg', '.txt')
        
        # 라벨 폴더가 없으면 생성
        os.makedirs(os.path.dirname(txt_path), exist_ok=True)
        
        with open(txt_path, 'a') as f: # 'a' 모드로 기존 라벨 뒤에 추가
            f.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
        print(f"✅ 새 박스 저장됨: {txt_path}")

    # --- 기존 이동/삭제 함수 ---
    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Space: self.next_image()
        elif event.key() == Qt.Key.Key_Delete: self.delete_sample()
        elif event.key() == Qt.Key.Key_Left: self.prev_image()

    def delete_sample(self):
        if not self.image_paths: return

        # 1. 삭제할 파일 경로 확보
        img_path = self.image_paths[self.current_idx]
        txt_path = img_path.replace(os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep).replace('.jpg', '.txt')

        # 2. 실제 파일 삭제 (이미지 + 라벨)
        try:
            if os.path.exists(img_path):
                os.remove(img_path)
                print(f"이미지 삭제 완료: {img_path}")
            
            if os.path.exists(txt_path):
                os.remove(txt_path)
                print(f"라벨 삭제 완료: {txt_path}")
        except Exception as e:
            print(f"삭제 중 오류 발생: {e}")

        # 3. 프로그램 내 목록에서 제거 (이게 없으면 다음 클릭 시 에러 발생)
        self.image_paths.pop(self.current_idx)

        # 4. 인덱스 조정 (마지막 사진을 지웠을 경우를 대비)
        if self.current_idx >= len(self.image_paths):
            self.current_idx = len(self.image_paths) - 1

        # 5. 다음 이미지 로드 (목록이 비었으면 종료 처리)
        if self.image_paths:
            self.load_image()
        else:
            self.label.clear()
            self.statusBar().showMessage("모든 이미지를 검수했습니다!")
            self.setWindowTitle("검수 완료")

    def next_image(self):
        if self.current_idx < len(self.image_paths) - 1:
            self.current_idx += 1
            self.load_image()

    def prev_image(self):
        if self.current_idx > 0:
            self.current_idx -= 1
            self.load_image()

    def resizeEvent(self, event):
        self.update_display()
        super().resizeEvent(event)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = YoloReviewer()
    ex.show()
    sys.exit(app.exec())