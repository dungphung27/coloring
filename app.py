# -*- coding: utf-8 -*-
import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import cv2
import base64
from io import BytesIO
from PIL import Image
from scipy.ndimage import binary_erosion
from skimage.color import label2rgb

from segmentation import segment_image_felzenszwalb

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = BASE_DIR

app = Flask(__name__, static_folder=STATIC_DIR)
CORS(app)

@app.after_request
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response

MAX_SIZE = 500


def decode_image(b64_string):
    if "," in b64_string:
        b64_string = b64_string.split(",")[1]
    img_bytes = base64.b64decode(b64_string)
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def encode_image(pil_img):
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode()


def resize_image(image):
    h, w = image.shape[:2]
    if max(h, w) > MAX_SIZE:
        ratio = MAX_SIZE / max(h, w)
        image = cv2.resize(image, (int(w * ratio), int(h * ratio)), interpolation=cv2.INTER_AREA)
    return image


def build_coloring_page(labels):
    h, w = labels.shape
    outline_img = Image.new("RGB", (w, h), "white")

    full_edge = np.zeros((h, w), dtype=bool)
    for lbl in np.unique(labels):
        mask = (labels == lbl)
        eroded = binary_erosion(mask)
        full_edge |= (mask & ~eroded)

    edge_array = np.array(outline_img)
    edge_array[full_edge] = [0, 0, 0]
    outline_img = Image.fromarray(edge_array)

    unique_labels = np.unique(labels)
    lbl_to_number = {lbl: i + 1 for i, lbl in enumerate(unique_labels)}

    return outline_img, lbl_to_number


def build_segmented_image(labels, image):
    h, w = labels.shape
    seg_uint8 = np.zeros((h, w, 3), dtype=np.uint8)

    unique_labels = np.unique(labels)
    lbl_to_number = {lbl: i + 1 for i, lbl in enumerate(unique_labels)}
    palette = {}

    for lbl in unique_labels:
        mask = (labels == lbl)
        # Tính màu trung bình thật sự từ ảnh gốc
        mean_color = image[mask].mean(axis=0).astype(np.uint8)
        seg_uint8[mask] = mean_color
        r, g, b = int(mean_color[0]), int(mean_color[1]), int(mean_color[2])
        palette[lbl_to_number[lbl]] = "#{:02X}{:02X}{:02X}".format(r, g, b)

    return Image.fromarray(seg_uint8), palette


@app.route("/favicon.ico")
def favicon():
    return "", 204

@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/api/segment", methods=["POST", "OPTIONS"])
def segment():
    if request.method == "OPTIONS":
        return "", 204

    data = request.get_json()
    if not data or "image" not in data:
        return jsonify({"error": "Thiếu ảnh"}), 400

    scale    = float(data.get("scale", 300.0))
    sigma    = float(data.get("sigma", 1.2))
    min_size = int(data.get("min_size", 500))

    try:
        image = decode_image(data["image"])
        image = resize_image(image)

        labels = segment_image_felzenszwalb(image, scale=scale, sigma=sigma, min_size=min_size)
        num_regions = len(np.unique(labels))

        segmented_pil, palette = build_segmented_image(labels, image)
        coloring_pil, _        = build_coloring_page(labels)
        original_pil           = Image.fromarray(image)

        # Remap label matrix về 0..N-1 liên tục
        unique_labels = np.unique(labels)
        lbl_remap = {lbl: i for i, lbl in enumerate(unique_labels)}
        remapped = np.vectorize(lbl_remap.get)(labels).astype(np.int32)

        return jsonify({
            "original":     encode_image(original_pil),
            "segmented":    encode_image(segmented_pil),
            "coloring":     encode_image(coloring_pil),
            "num_regions":  num_regions,
            "palette":      palette,
            "label_matrix": remapped.tolist(),   # list of lists (H x W)
            "img_width":    int(remapped.shape[1]),
            "img_height":   int(remapped.shape[0]),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)
