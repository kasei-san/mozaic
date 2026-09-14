"""ComfyUI 用の自動モザイクノード。VAE Decode と Save Image の間に挟む。

検出とモザイクの処理はリポジトリ直下の auto_mosaic.py にあり、このノードはそれを呼んで
ComfyUI の画像テンソルと受け渡しするだけ（処理を二重に持たない）。
配置方法は docs/auto_mosaic.md の「ComfyUI で使う」を参照。
"""
import importlib.util
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import folder_paths


def _load_core():
    """auto_mosaic.py を探す。①このフォルダ ②環境変数 MOZAIC_DIR ③リンク元のリポジトリ直下。"""
    here = Path(__file__).resolve().parent
    candidates = [here / "auto_mosaic.py"]
    if os.environ.get("MOZAIC_DIR"):
        candidates.append(Path(os.environ["MOZAIC_DIR"]) / "auto_mosaic.py")
    candidates.append(here.parents[1] / "auto_mosaic.py")
    for p in candidates:
        if p.is_file():
            spec = importlib.util.spec_from_file_location("mozaic_auto_core", p)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    raise ImportError("auto_mosaic.py が見つかりません: " + ", ".join(str(p) for p in candidates))


core = _load_core()


def _ultralytics_dirs() -> list:
    """models/ultralytics（ComfyUI-Impact-Pack と同じ置き場所）の候補。
    extra_model_paths.yaml でモデルを別の場所に置いている環境もあるので、checkpoints の隣も見る。"""
    dirs = []
    if "ultralytics" in folder_paths.folder_names_and_paths:
        dirs += [Path(p) for p in folder_paths.folder_names_and_paths["ultralytics"][0]]
    dirs.append(Path(folder_paths.models_dir) / "ultralytics")
    dirs += [Path(p).parent / "ultralytics" for p in folder_paths.get_folder_paths("checkpoints")]
    seen, out = set(), []
    for d in dirs:
        key = str(d.resolve()).lower()
        if key not in seen and d.is_dir():
            seen.add(key)
            out.append(d)
    return out


def _model_names() -> list:
    names = []
    for base in _ultralytics_dirs():
        for sub in ("segm", "bbox"):
            names += [f"{sub}/{p.name}" for p in sorted((base / sub).glob("*.pt"))]
    return sorted(set(names))


def _model_file(name: str) -> Path:
    for base in _ultralytics_dirs():
        if (base / name).is_file():
            return base / name
    raise FileNotFoundError(f"モデルが見つかりません: models/ultralytics/{name}")


class MozaicAutoMosaic:
    @classmethod
    def INPUT_TYPES(cls):
        names = _model_names() or ["(models/ultralytics に .pt がありません)"]
        default = next((n for n in names if n.endswith("animeNSFWSegm_xlRes1280.pt")), names[0])
        return {"required": {
            "image": ("IMAGE",),
            "model_1": (names, {"default": default}),
            "model_2": (["none"] + names, {"default": "none"}),
            "model_3": (["none"] + names, {"default": "none"}),
            "model_4": (["none"] + names, {"default": "none"}),
            "confidence": ("FLOAT", {"default": core.DEFAULT_CONF, "min": 0.05, "max": 0.95, "step": 0.05}),
            "coverage_expansion": ("INT", {"default": core.DEFAULT_EXPAND, "min": 0, "max": 128,
                                           "tooltip": "マスクを外へ広げる幅（px）"}),
            "grain_size": ("INT", {"default": 0, "min": 0, "max": 128,
                                   "tooltip": "モザイクの1マス（px）。0 = 画像幅の1/100（mozaic と同じ）"}),
            "classes": ("STRING", {"default": ",".join(core.DEFAULT_CLASSES)}),
        }}

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "run"
    CATEGORY = "mozaic"

    def run(self, image, model_1, model_2, model_3, model_4, confidence, coverage_expansion, grain_size, classes):
        models = [_model_file(m) for m in (model_1, model_2, model_3, model_4) if m and m != "none"]
        cls = [c.strip() for c in classes.split(",") if c.strip()]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        images, masks = [], []
        for t in image:                                       # バッチの1枚ずつ
            im = Image.fromarray((t.cpu().numpy() * 255).clip(0, 255).astype(np.uint8))
            res, mask, hits = core.apply(im, models, confidence, coverage_expansion, grain_size, cls, device)
            print(f"[mozaic auto] 検出 {len(hits)} 件 {hits[:4]}")
            images.append(torch.from_numpy(np.array(res).astype(np.float32) / 255.0))
            if not hits:
                mask = Image.new("L", im.size, 0)
            masks.append(torch.from_numpy(np.array(mask).astype(np.float32) / 255.0))
        return (torch.stack(images), torch.stack(masks))


NODE_CLASS_MAPPINGS = {"MozaicAutoMosaic": MozaicAutoMosaic}
NODE_DISPLAY_NAME_MAPPINGS = {"MozaicAutoMosaic": "Auto Mosaic (YOLO SEGM)"}
