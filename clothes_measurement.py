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
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 明暗反転の適応的二値化で照明変化に強くする
    thresh = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        11,
        2,
    )

    # 形態学的処理でノイズ除去＆穴埋め
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

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
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        if len(approx) == 4 and cv2.isContourConvex(approx):  # 頂点が4つの凸多角形
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)
            fill_ratio = area / float(w * h)

            # 正方形かつ内部が詰まっているか
            if 0.9 <= aspect_ratio <= 1.1 and fill_ratio > 0.8 and area > max_area:
                max_area = area
                best_cnt = approx

    # マーカーが見つかった場合
    if best_cnt is not None:
        x, y, w, h = cv2.boundingRect(best_cnt)
        cm_per_pixel = marker_size_cm / np.mean([w, h])

        # デバッグ描画
        cv2.drawContours(image, [best_cnt], -1, (0, 0, 255), 2)
        cv2.putText(
            image,
            "Marker",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )

    return cm_per_pixel


# 背景除去
def remove_background(image):
    result = remove(image)
    return cv2.cvtColor(np.array(result), cv2.COLOR_RGBA2BGR)

# 服計測
def measure_clothes(image, cm_per_pixel):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, {}, {}

    clothes_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(clothes_contour)

    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [clothes_contour], -1, 255, -1)

    def horizontal_measure(line_y):
        cols = np.where(mask[line_y] > 0)[0]
        if cols.size == 0:
            return 0, (0, line_y, 0, line_y)
        left, right = cols.min(), cols.max()
        return right - left, (left, line_y, right, line_y)

    # 肩幅（上から10%の位置での幅）
    shoulder_y = y + int(h * 0.1)
    shoulder_pixels, shoulder_line = horizontal_measure(shoulder_y)

    # 身幅（胸あたり＝上から30%）
    chest_y = y + int(h * 0.3)
    chest_pixels, chest_line = horizontal_measure(chest_y)

    # 身丈（最上部から最下部まで）
    body_pixels = h
    body_line = (x + w // 2, y, x + w // 2, y + h)

    # 袖丈（肩幅と身幅の差から推定）
    sleeve_pixels = (shoulder_pixels - chest_pixels) / 2

    measures = {
        "肩幅": shoulder_pixels * cm_per_pixel,
        "身幅": chest_pixels * cm_per_pixel,
        "身丈": body_pixels * cm_per_pixel,
        "袖丈": sleeve_pixels * cm_per_pixel,
    }

    lines = {
        "肩幅": shoulder_line,
        "身幅": chest_line,
        "身丈": body_line,
    }

    return clothes_contour, measures, lines

# 画像に寸法描画
def draw_measurements_on_image(image, measurements, lines):
    font = cv2.FONT_HERSHEY_SIMPLEX
    # 線とテキストを描画
    for key, line in lines.items():
        x1, y1, x2, y2 = line
        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        tx = int((x1 + x2) / 2)
        ty = int((y1 + y2) / 2) - 10
        cv2.putText(
            image,
            f"{key}: {measurements[key]:.1f} cm",
            (tx, ty),
            font,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    # 線を引かない寸法（袖丈など）は左上にまとめて表示
    y_offset = 30
    for key, value in measurements.items():
        if key in lines:
            continue
        text = f"{key}: {value:.1f} cm"
        cv2.putText(
            image,
            text,
            (30, y_offset),
            font,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        y_offset += 30
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
    contour, measurements, lines = measure_clothes(img_no_bg, cm_per_pixel)
    if contour is None:
        print("服が検出できません。")
        exit()

    # 寸法表示
    for k, v in measurements.items():
        print(f"{k}: {v:.1f} cm")

    img_with_text = draw_measurements_on_image(img.copy(), measurements, lines)
    cv2.drawContours(img_with_text, [contour], -1, (255, 0, 0), 2)

    # 保存
    cv2.imwrite("clothes_with_measurements.jpg", img_with_text)
    print("寸法入り画像を保存しました → clothes_with_measurements.jpg")
