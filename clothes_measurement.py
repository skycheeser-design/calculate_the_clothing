import cv2
import numpy as np
from PIL import Image
import pillow_heif
import os
from rembg import remove

# HEIC対応読み込み
def load_image(path):
    ext = os.path.splitext(path)[-1].lower()
    if ext == ".heic":
        heif_file = pillow_heif.read_heif(path)
        img = Image.frombytes(heif_file.mode, heif_file.size, heif_file.data, "raw")
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    else:
        return cv2.imread(path)

# 黒四角マーカー検出
def detect_marker(image, marker_size_cm=5.0):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 明暗反転の二値化（黒マーカーが白になるように）
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)

    # 輪郭抽出
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_cnt = None
    max_area = 0
    cm_per_pixel = None

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 100:  # 小さすぎるノイズ除去
            continue

        # 四角形近似
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)

        if len(approx) == 4:  # 頂点数が4つ
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)

            # 正方形っぽいか確認
            if 0.95 <= aspect_ratio <= 1.05 and area > max_area:
                max_area = area
                best_cnt = cnt

    # マーカーが見つかった場合
    if best_cnt is not None:
        x, y, w, h = cv2.boundingRect(best_cnt)
        cm_per_pixel = marker_size_cm / np.mean([w, h])

        # デバッグ描画
        cv2.drawContours(image, [best_cnt], -1, (0, 0, 255), 2)
        cv2.putText(image, "Marker", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    return cm_per_pixel


# 背景除去
def remove_background(image):
    result = remove(image)
    return cv2.cvtColor(np.array(result), cv2.COLOR_RGBA2BGR)

# 服計測
def measure_clothes(image, cm_per_pixel):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, {}

    clothes_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(clothes_contour)

    # 肩幅（上から10%の位置での幅）
    shoulder_y = y + int(h * 0.1)
    shoulder_line = gray[shoulder_y:shoulder_y+5, x:x+w]
    shoulder_width = np.max(np.where(shoulder_line < 128)) - np.min(np.where(shoulder_line < 128))

    # 身幅（胸あたり＝上から30%）
    chest_y = y + int(h * 0.3)
    chest_line = gray[chest_y:chest_y+5, x:x+w]
    chest_width = np.max(np.where(chest_line < 128)) - np.min(np.where(chest_line < 128))

    # 袖丈（肩端から袖端まで）
    sleeve_length = (shoulder_width - chest_width) / 2

    # 身丈
    body_length = h

    measures = {
        "肩幅": shoulder_width * cm_per_pixel,
        "身幅": chest_width * cm_per_pixel,
        "身丈": body_length * cm_per_pixel,
        "袖丈": sleeve_length * cm_per_pixel
    }
    return clothes_contour, measures

# 画像に寸法描画
def draw_measurements_on_image(image, measurements):
    font = cv2.FONT_HERSHEY_SIMPLEX
    y_offset = 30
    for key, value in measurements.items():
        text = f"{key}: {value:.1f} cm"
        cv2.putText(image, text, (30, y_offset),
                    font, 1, (0, 255, 0), 2, cv2.LINE_AA)
        y_offset += 40
    return image


if __name__ == "__main__":
    image_path = "image.jpg"  # HEICもJPEGもOK
    img = load_image(image_path)

    # マーカー検出（背景除去前）
    cm_per_pixel = detect_marker(img)
    if cm_per_pixel is None:
        print("マーカーが検出できません。終了します。")
        exit()

    # 背景除去
    img_no_bg = remove_background(img)

    # 服計測
    contour, measurements = measure_clothes(img_no_bg, cm_per_pixel)
    if contour is None:
        print("服が検出できません。")
        exit()

    # 寸法表示
    for k, v in measurements.items():
        print(f"{k}: {v:.1f} cm")

    img_with_text = draw_measurements_on_image(img.copy(), measurements)
    cv2.drawContours(img_with_text, [contour], -1, (255, 0, 0), 2)

    # 保存
    cv2.imwrite("clothes_with_measurements.jpg", img_with_text)
    print("寸法入り画像を保存しました → clothes_with_measurements.jpg")
