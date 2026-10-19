"""アニメ絵の局部検出（deepghs/anime_censor_detection の YOLOv8 ONNX を直接使う）

dghs-imgutils 経由だと pandas・scipy・scikit-learn・opencv まで入り、numpy も 1.26 に
下がるので、モデルだけ取ってきて前処理と後処理を numpy で書いている。
"""
import hashlib
import os
import shutil
import urllib.request

import numpy as np
from PIL import Image

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "censor_detect_v1.0_s.onnx")
# リビジョンを固定し、sha256 で照合する（ライセンスは MIT）
MODEL_URL = (
    "https://huggingface.co/deepghs/anime_censor_detection/resolve/"
    "0cf62fd6b28213b40ae0c0055f92e7ae6a96bdc2/censor_detect_v1.0_s/model.onnx"
)
MODEL_SHA256 = "2c2524824d7d320c5619a0a73702a2e2186f619067c823d211e94f7cfe489cba"
LABELS = ["nipple_f", "penis", "pussy"]

INFER_MAX = 640   # 長辺をこの長さに縮めて推論する（学習時の解像度）
STRIDE = 32       # 入力の縦横はこの倍数でなければならない
IOU_THRESHOLD = 0.7
DOWNLOAD_TIMEOUT_SEC = 30

_session = None


def is_available():
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return False
    return True


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _ensure_model():
    if os.path.exists(MODEL_PATH):
        # 既存のファイルも照合する（43MB で 0.1 秒程度）。壊れていたら消して取り直させる
        if _sha256(MODEL_PATH) == MODEL_SHA256:
            return
        os.remove(MODEL_PATH)
        raise RuntimeError("models/ のモデルが壊れていたので消したのだ。もう一度押すと取り直すのだ")
    os.makedirs(MODEL_DIR, exist_ok=True)
    tmp = MODEL_PATH + ".part"
    # urlretrieve はタイムアウトを持てず、回線が止まるとワーカーが戻らない
    # （ボタンが押せないままになる）。timeout は1回の読み取りの待ち時間
    with urllib.request.urlopen(MODEL_URL, timeout=DOWNLOAD_TIMEOUT_SEC) as res, \
            open(tmp, "wb") as f:
        shutil.copyfileobj(res, f, 1 << 20)
    if _sha256(tmp) != MODEL_SHA256:
        os.remove(tmp)
        raise RuntimeError("ダウンロードしたモデルの sha256 が一致しないのだ")
    os.replace(tmp, MODEL_PATH)


def _get_session():
    global _session
    if _session is None:
        import onnxruntime as ort
        _ensure_model()
        _session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    return _session


def _nms(boxes, scores, iou_threshold):
    """boxes: (N, 4) x1y1x2y2。スコア順に重なりの大きいものを落とす"""
    order = scores.argsort()[::-1]
    keep = []
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    while order.size:
        i = order[0]
        keep.append(i)
        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
        order = rest[iou <= iou_threshold]
    return keep


def detect(image, conf_threshold, labels):
    """image: RGB の PIL 画像。[(x1, y1, x2, y2, label, score), ...] を元画像の座標で返す

    初回はモデルのダウンロード（約 43MB）とロードが走るので、メインスレッドから呼ばないこと。
    """
    session = _get_session()
    iw, ih = image.size
    ratio = INFER_MAX / max(iw, ih)
    # 縦横それぞれ STRIDE の倍数に丸める。縦横比は僅かにずれるので倍率は別々に持つ
    tw = max(STRIDE, int(round(iw * ratio / STRIDE)) * STRIDE)
    th = max(STRIDE, int(round(ih * ratio / STRIDE)) * STRIDE)
    arr = np.asarray(image.resize((tw, th), Image.BILINEAR), dtype=np.float32) / 255.0
    x = arr.transpose(2, 0, 1)[np.newaxis]

    out = session.run(None, {session.get_inputs()[0].name: x})[0][0].T  # (N, 4 + クラス数)
    # 対象外のクラスを先に 0 にしてから argmax する。逆順だと、対象外（例: nipple_f）が
    # 最大のアンカーは、対象のクラスがしきい値を超えていても捨てられて見落としになる
    class_scores = out[:, 4:].copy()
    wanted = np.isin(np.arange(len(LABELS)), [LABELS.index(n) for n in labels if n in LABELS])
    class_scores[:, ~wanted] = 0.0
    cls = class_scores.argmax(axis=1)
    score = class_scores[np.arange(len(cls)), cls]
    sel = score >= conf_threshold
    if not sel.any():
        return []
    cx, cy, w, h = out[sel, 0], out[sel, 1], out[sel, 2], out[sel, 3]
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    cls, score = cls[sel], score[sel]

    results = []
    sx, sy = iw / tw, ih / th
    for c in np.unique(cls):  # NMS はクラスごと
        idx = np.where(cls == c)[0]
        for k in _nms(boxes[idx], score[idx], IOU_THRESHOLD):
            x1, y1, x2, y2 = (float(v) for v in boxes[idx[k]])
            results.append((
                max(0.0, x1 * sx), max(0.0, y1 * sy),
                min(float(iw), x2 * sx), min(float(ih), y2 * sy),
                LABELS[c], float(score[idx[k]]),
            ))
    return results
