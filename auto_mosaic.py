"""自動モザイク: YOLO のセグメンテーションモデルで局部を検出し、その範囲だけにモザイクをかける。

    venv\\Scripts\\python auto_mosaic.py <元画像フォルダ>                 # 結果を input/ へ
    venv\\Scripts\\python auto_mosaic.py <元画像フォルダ> --out done --misses input
    venv\\Scripts\\python auto_mosaic.py <元画像フォルダ> --measure       # 数えるだけ（書き出さない）

処理は 検出 → 範囲を広げる → モザイク画像を作る → マスクで合成 の4段。
モザイクの粒度は app.py と同じ（画像幅の 1/100・最小 4px）。
使い方の詳細は docs/auto_mosaic.md、実装上の注意は CLAUDE.md の「自動モザイク」を参照。
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
INPUT_DIR = HERE / "input"
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")

# app.py と同じ値（変えるときは app.py も揃える）
MOSAIC_BLOCK_RATIO = 100
MOSAIC_BLOCK_MIN = 4
JPEG_QUALITY = 95

DEFAULT_MODELS = ("models/segm/animeNSFWSegm_xlRes1280.pt",)
# 隠す部位のクラス名。モデルごとに呼び名が違うので、docs に載せたモデルの呼び名を全部並べる
# （乳首・顔・陰毛は隠さない）。モデルに無いクラス名は無視される
DEFAULT_CLASSES = ("penis", "vagina", "anus", "pussy", "genitalia_f")
DEFAULT_CONF = 0.35     # 検出の信頼度の下限
DEFAULT_EXPAND = 30     # マスクを外へ広げる幅（px）。輪郭ぴったりだと縁が残る

_models: dict[str, object] = {}


def model_path(name: str | os.PathLike) -> Path:
    """相対パスはこのファイルのあるフォルダから解決する。"""
    p = Path(name)
    return p if p.is_absolute() else HERE / p


def load_model(name):
    from ultralytics import YOLO

    key = str(model_path(name))
    if key not in _models:
        _models[key] = YOLO(key)
    return _models[key]


def pixelate(im, block: int | None = None):
    """画像全体をモザイクにした画像を返す。block を省くと app.py と同じ粒度。"""
    from PIL import Image

    w, h = im.size
    b = block or max(MOSAIC_BLOCK_MIN, w // MOSAIC_BLOCK_RATIO)
    bw, bh = max(1, w // b), max(1, h // b)
    return im.resize((bw, bh), Image.BOX).resize((w, h), Image.NEAREST)


def detect_mask(im, models=DEFAULT_MODELS, conf: float = DEFAULT_CONF, classes=DEFAULT_CLASSES, device="cpu"):
    """検出した範囲を白で塗ったマスク（L モード）と、検出の一覧 [(モデル, クラス, 信頼度)] を返す。

    セグメンテーションモデルは形のとおりに、枠（bbox）しか出さないモデルは枠を塗る。
    複数モデルの結果は足し合わせる。
    """
    import numpy as np
    from PIL import Image

    w, h = im.size
    union = np.zeros((h, w), dtype=bool)
    hits = []
    want = {c.lower() for c in classes}
    for name in models:
        m = load_model(name)
        ids = [i for i, n in m.names.items() if str(n).lower() in want]
        if not ids:
            continue
        r = m.predict(im, conf=conf, classes=ids, device=device, verbose=False)[0]
        for k, (c, s) in enumerate(zip(r.boxes.cls.tolist(), r.boxes.conf.tolist())):
            hits.append((Path(name).name, m.names[int(c)], round(float(s), 2)))
            if r.masks is not None:
                mk = Image.fromarray((r.masks.data[k].cpu().numpy() * 255).astype("uint8")).resize((w, h))
                union |= np.array(mk) > 127
            else:
                x1, y1, x2, y2 = (int(v) for v in r.boxes.xyxy[k].tolist())
                union[max(0, y1):y2, max(0, x1):x2] = True
    return Image.fromarray((union * 255).astype("uint8")), hits


def expand_mask(mask, px: int):
    """マスクを外へ px 広げる。"""
    if px <= 0:
        return mask
    try:
        import cv2
        import numpy as np
        from PIL import Image

        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
        return Image.fromarray(cv2.dilate(np.array(mask), k))
    except ImportError:
        from PIL import ImageFilter

        return mask.filter(ImageFilter.MaxFilter(2 * px + 1))


def apply(im, models=DEFAULT_MODELS, conf: float = DEFAULT_CONF, expand_px: int = DEFAULT_EXPAND,
          grain: int = 0, classes=DEFAULT_CLASSES, device="cpu"):
    """(モザイク後の画像, 使ったマスク, 検出の一覧) を返す。検出ゼロなら画像はそのまま返す。"""
    from PIL import Image

    im = im.convert("RGB")
    mask, hits = detect_mask(im, models, conf, classes, device)
    if not hits:
        return im, mask, hits
    mask = expand_mask(mask, expand_px)
    return Image.composite(pixelate(im, grain or None), im, mask), mask, hits


def list_images(folder: Path) -> list[Path]:
    """app.py と同じく、フォルダ直下の画像だけを名前順に返す（サブフォルダは見ない）。"""
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="YOLO で局部を検出して、その範囲だけにモザイクをかける")
    ap.add_argument("src", help="元画像のフォルダ（直下の画像だけを処理する）")
    ap.add_argument("--out", default=str(INPUT_DIR), help="書き出し先（既定: input/）")
    ap.add_argument("--misses", default="", help="検出ゼロの画像のコピー先（既定: --out と同じ）")
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS), help="モデルのパス。カンマ区切りで複数（最大4本）")
    ap.add_argument("--conf", type=float, default=DEFAULT_CONF, help=f"信頼度の下限（既定 {DEFAULT_CONF}）")
    ap.add_argument("--expand", type=int, default=DEFAULT_EXPAND, help=f"マスクを広げる幅 px（既定 {DEFAULT_EXPAND}）")
    ap.add_argument("--grain", type=int, default=0, help="モザイクの1マス px（既定 0 = app.py と同じ粒度）")
    ap.add_argument("--classes", default=",".join(DEFAULT_CLASSES), help="隠すクラス名（カンマ区切り）")
    ap.add_argument("--device", default="cpu", help="cpu / 0（1枚目のGPU）など")
    ap.add_argument("--overwrite", action="store_true", help="書き出し先に同名ファイルがあっても上書きする")
    ap.add_argument("--measure", action="store_true", help="検出して数えるだけ（何も書き出さない）")
    ap.add_argument("--report", default="", help="検出の一覧を JSON で保存するパス")
    a = ap.parse_args()

    try:
        import ultralytics  # noqa: F401
    except ImportError:
        print("ultralytics が入っていません。先に次を実行してください:\n"
              "  venv\\Scripts\\pip install -r requirements-auto.txt")
        return 1
    from PIL import Image

    src = Path(a.src).resolve()
    out = Path(a.out).resolve()
    misses_dir = Path(a.misses).resolve() if a.misses else out
    if not src.is_dir():
        print(f"フォルダが見つかりません: {src}")
        return 1
    if not a.measure and src in (out, misses_dir):
        print("元画像のフォルダと書き出し先が同じです。元画像を上書きしないよう、別のフォルダを指定してください")
        return 1
    models = [m.strip() for m in a.models.split(",") if m.strip()][:4]
    missing = [m for m in models if not model_path(m).is_file()]
    if missing:
        print("モデルファイルが見つかりません:\n" + "\n".join(f"  {model_path(m)}" for m in missing)
              + "\n入手先と置き場所は docs/auto_mosaic.md の「セットアップ」を参照してください")
        return 1
    classes = [c.strip() for c in a.classes.split(",") if c.strip()]
    files = list_images(src)
    if not files:
        print(f"画像がありません: {src}")
        return 1

    rows, hit_names, miss_names, skipped = [], [], [], []
    for i, f in enumerate(files, 1):
        # 前回の結果（手で塗り足したものを含む）を潰さない。行き先は検出の有無で変わるので両方を見る
        if not a.measure and not a.overwrite and ((out / f.name).exists() or (misses_dir / f.name).exists()):
            skipped.append(f.name)
            print(f"  [{i}/{len(files)}] スキップ（書き出し先に同名あり）  {f.name}", flush=True)
            continue
        with Image.open(f) as im0:
            if a.measure:
                _, hits = detect_mask(im0.convert("RGB"), models, a.conf, classes, a.device)
            else:
                img, _, hits = apply(im0, models, a.conf, a.expand, a.grain, classes, a.device)
                dst = (out if hits else misses_dir) / f.name
                dst.parent.mkdir(parents=True, exist_ok=True)
                if not hits:
                    shutil.copy2(f, dst)            # 加工していない画像は再保存せずバイナリコピー（JPEG を劣化させない）
                elif f.suffix.lower() in (".jpg", ".jpeg"):
                    img.save(dst, quality=JPEG_QUALITY)
                else:
                    img.save(dst)
        rows.append({"file": f.name, "hits": hits})
        (hit_names if hits else miss_names).append(f.name)
        print(f"  [{i}/{len(files)}] {'検出 ' + str(len(hits)) + ' 件' if hits else '検出ゼロ'}  {f.name}", flush=True)

    n = len(files)
    print(f"\n対象 {n} 枚 — 検出あり {len(hit_names)} 枚 / 検出ゼロ {len(miss_names)} 枚 / スキップ {len(skipped)} 枚")
    if not a.measure:
        print(f"  モザイク済み → {out}")
        print(f"  検出ゼロ（加工せずコピー） → {misses_dir}")
        if skipped:
            print("  スキップは書き出し先に同名があった画像です（処理し直すなら --overwrite）")
    for name in miss_names:
        print(f"  検出ゼロ: {name}")
    print("\n※ モザイクがかかるのは検出できた範囲だけです。検出ありの画像でも、写っている部位の全部に"
          "かかったとは限りません。run.bat で1枚ずつ確認して、漏れはブラシで塗ってください")
    if a.report:
        Path(a.report).write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
