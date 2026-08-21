import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk, ImageDraw, ImageFilter
import numpy as np
from collections import deque
import io
import os
import json
import queue
import shutil
import threading

INPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "input")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
SETTINGS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "settings.json")
IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')
MOSAIC_BLOCK_RATIO = 100
MOSAIC_BLOCK_MIN = 4
# 高さは全ウィジェットが収まる実測値（長いファイル名でステータスが折り返した状態で 841）
# に余裕を持たせた値。足りなくなると Undo ボタンから先に隠れる
# 幅は左のコントロール 160 + 右のサムネイル一覧 + 編集領域
WINDOW_W, WINDOW_H = 1440, 880
WINDOW_SCREEN_MARGIN_W, WINDOW_SCREEN_MARGIN_H = 80, 100
THUMB_W = 116           # サムネイルの最大辺
THUMB_CELL_H = 128      # 一覧1セルの高さ（サムネイルの縦横比によらず固定）
THUMB_PANEL_W = 150     # スクロールバーを含む右パネルの幅
JPEG_QUALITY = 95
SAVE_DELAY_MS = 400     # 連続ストロークをまとめる待ち時間
BRUSH_SIZE_MIN, BRUSH_SIZE_MAX = 3, 50
WAND_MIN, WAND_MAX = 1, 100
BRUSH_SHAPES = [("○", "circle"), ("□", "square")]
ZOOM_FACTOR = 1.15
ZOOM_MIN = 0.05
ZOOM_MAX = 20.0

_DEFAULT_SETTINGS = {
    "wand_tolerance": 50,
    "wand_dilate_scale": 50,
    "line_threshold": 80,
    "line_dilate_scale": 25,
    "brush_shape": "circle",
    "brush_size": 20,
}

def _load_settings():
    try:
        with open(SETTINGS_FILE, encoding="utf-8") as f:
            data = json.load(f)
        settings = {**_DEFAULT_SETTINGS, **data}
    except (FileNotFoundError, json.JSONDecodeError):
        settings = dict(_DEFAULT_SETTINGS)
    if settings["brush_shape"] not in [s for _, s in BRUSH_SHAPES]:
        settings["brush_shape"] = _DEFAULT_SETTINGS["brush_shape"]
    try:
        size = int(settings["brush_size"])
    except (TypeError, ValueError):
        size = _DEFAULT_SETTINGS["brush_size"]
    settings["brush_size"] = max(BRUSH_SIZE_MIN, min(size, BRUSH_SIZE_MAX))
    for key in ("wand_tolerance", "wand_dilate_scale"):
        try:
            value = int(settings[key])
        except (TypeError, ValueError):
            value = _DEFAULT_SETTINGS[key]
        settings[key] = max(WAND_MIN, min(value, WAND_MAX))
    return settings

SETTINGS = _load_settings()


class MosaicApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Mosaic Tool")
        # 画面より大きくならないようクランプする（タイトルバー・タスクバーの分を引く）
        win_w = min(WINDOW_W, root.winfo_screenwidth() - WINDOW_SCREEN_MARGIN_W)
        win_h = min(WINDOW_H, root.winfo_screenheight() - WINDOW_SCREEN_MARGIN_H)
        self.root.geometry(f"{win_w}x{win_h}")
        self.root.configure(bg="#1e1e1e")

        # entries: [(ファイル名, 元画像のパス), ...]。実体は _build_entries を見ること
        self.entries = []
        self.current_index = 0
        # ファイル名 -> PNG 圧縮したマスク。素の L 画像だと 1024x1024 で 1MB/枚 だが、
        # 2値のベタ塗りなので PNG にすると 6KB 程度に落ちる（実測 約180分の1）
        self.masks = {}
        # このセッションで一度でも書き戻したファイル。Undo でマスクが空に戻ったとき、
        # 塗る前の状態を output へ書き戻す必要があるかの判定に使う
        self.saved = set()
        self.copy_errors = []

        # tool: "brush" | "eraser" | "wand" | "line"
        self.tool = "brush"
        self.brush_size = SETTINGS["brush_size"]
        # brush_shape: "circle" | "square"（ブラシ・消しゴム・黒線ブラシ共通）
        self.brush_shape = SETTINGS["brush_shape"]

        self.original_image = None
        self.pixelated_image = None
        self.mask_image = None
        self.composite_image = None
        self.tk_image = None
        self._img_item = None

        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0

        self._pan_start = None
        self._pan_offset_start = (0, 0)
        self._cursor_pos = None
        self._drawing = False  # ブラシドラッグ中

        self.wand_tolerance = SETTINGS["wand_tolerance"]
        self.wand_dilate_scale = SETTINGS["wand_dilate_scale"]
        self.line_threshold = SETTINGS["line_threshold"]
        self.line_dilate_scale = SETTINGS["line_dilate_scale"]

        # Undo: (ファイル名, PNG圧縮したマスク) のスタック。画像ごとではなく全画像で1本
        self.undo_stack = []

        # サムネイル一覧の描画物（GC されないよう PhotoImage を保持する）
        self.thumb_photos = {}
        self.thumb_items = {}
        self.thumb_marks = {}
        self.thumb_sel = None
        self._thumb_queue = queue.Queue()
        self._thumb_done = False
        self._save_after_id = None

        self._build_entries()
        self._setup_ui()

        if not self.entries:
            # 一覧は output を見るので、input が空でも output に残っていれば起動する
            messagebox.showwarning("警告", "input にも output にも画像がないのだ")
            root.destroy()
            return
        if self.copy_errors:
            messagebox.showwarning(
                "警告",
                "output にコピーできなかった画像があるのだ:\n\n"
                + "\n".join(self.copy_errors[:10])
                + ("\n..." if len(self.copy_errors) > 10 else ""),
            )

        self._build_thumb_cells()
        self._select_index(0)
        self._start_thumb_loader()

    # ── UI構築 ──────────────────────────────────────────────

    def _setup_ui(self):
        ctrl = tk.Frame(self.root, width=160, bg="#2b2b2b")
        ctrl.pack(side=tk.LEFT, fill=tk.Y)
        ctrl.pack_propagate(False)

        # ステータスは下端固定。pack は宣言順に領域を切り出すので、ウィンドウが低いときでも
        # 確実に表示されるよう他より先に pack する
        # （side=BOTTOM を指定しても、後から pack すると領域が残っておらず消える）
        self.status_var = tk.StringVar()
        tk.Label(ctrl, textvariable=self.status_var,
                 bg="#2b2b2b", fg="#888888",
                 wraplength=148, justify=tk.CENTER).pack(side=tk.BOTTOM, pady=16, padx=6)

        tk.Label(ctrl, text="← → ↑ ↓  で画像を移動\n塗ると自動保存",
                 bg="#2b2b2b", fg="#666666",
                 font=("", 8), justify=tk.CENTER).pack(side=tk.BOTTOM, pady=(0, 4))

        tk.Frame(ctrl, height=1, bg="#444").pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=14)

        tk.Label(ctrl, text="筆の太さ", bg="#2b2b2b", fg="#cccccc",
                 font=("", 10, "bold")).pack(pady=(20, 0))

        self.brush_var = tk.IntVar(value=self.brush_size)
        tk.Scale(
            ctrl, from_=BRUSH_SIZE_MIN, to=BRUSH_SIZE_MAX, orient=tk.HORIZONTAL,
            variable=self.brush_var,
            command=lambda v: self._set_brush(int(v)),
            bg="#2b2b2b", fg="#cccccc", highlightthickness=0,
            troughcolor="#444", activebackground="#777", length=136,
        ).pack(padx=12)

        tk.Label(ctrl, text="筆の形", bg="#2b2b2b", fg="#cccccc",
                 font=("", 10, "bold")).pack(pady=(8, 4))

        shape_row = tk.Frame(ctrl, bg="#2b2b2b")
        shape_row.pack()
        self.shape_var = tk.StringVar(value=self.brush_shape)
        for label, shape in BRUSH_SHAPES:
            rb = tk.Radiobutton(
                shape_row, text=label,
                variable=self.shape_var, value=shape,
                bg="#2b2b2b", fg="#cccccc", font=("", 12),
                selectcolor="#444444", activebackground="#2b2b2b",
                activeforeground="#ffffff",
                command=lambda s=shape: self._set_brush_shape(s),
            )
            rb.pack(side=tk.LEFT, padx=4)

        tk.Frame(ctrl, height=1, bg="#444").pack(fill=tk.X, padx=10, pady=14)

        self.tool_buttons = {}
        for label, tool in [("ブラシ", "brush"), ("消しゴム", "eraser"),
                            ("魔法の杖", "wand"), ("黒線ブラシ", "line")]:
            btn = tk.Button(
                ctrl, text=label,
                command=lambda t=tool: self._set_tool(t),
                bg="#555555", fg="white", relief=tk.RAISED,
                activebackground="#777", bd=0, padx=8, pady=6,
            )
            btn.pack(fill=tk.X, padx=12, pady=4)
            self.tool_buttons[tool] = btn

        tk.Frame(ctrl, height=1, bg="#444").pack(fill=tk.X, padx=10, pady=(10, 4))
        tk.Label(ctrl, text="色差 tolerance", bg="#2b2b2b", fg="#aaaaaa",
                 font=("", 9)).pack(anchor=tk.W, padx=12)
        self.tolerance_var = tk.IntVar(value=self.wand_tolerance)
        tk.Scale(
            ctrl, from_=WAND_MIN, to=WAND_MAX, orient=tk.HORIZONTAL,
            variable=self.tolerance_var,
            command=lambda v: setattr(self, "wand_tolerance", int(v)),
            bg="#2b2b2b", fg="#cccccc", highlightthickness=0,
            troughcolor="#444", activebackground="#777", length=136,
        ).pack(padx=12)

        tk.Label(ctrl, text="境界膨張", bg="#2b2b2b", fg="#aaaaaa",
                 font=("", 9)).pack(anchor=tk.W, padx=12, pady=(6, 0))
        self.dilate_var = tk.IntVar(value=self.wand_dilate_scale)
        tk.Scale(
            ctrl, from_=WAND_MIN, to=WAND_MAX, orient=tk.HORIZONTAL,
            variable=self.dilate_var,
            command=lambda v: setattr(self, "wand_dilate_scale", int(v)),
            bg="#2b2b2b", fg="#cccccc", highlightthickness=0,
            troughcolor="#444", activebackground="#777", length=136,
        ).pack(padx=12)

        tk.Frame(ctrl, height=1, bg="#444").pack(fill=tk.X, padx=10, pady=(10, 4))
        tk.Label(ctrl, text="黒線ブラシ設定", bg="#2b2b2b", fg="#cccccc",
                 font=("", 9, "bold")).pack(anchor=tk.W, padx=12)
        tk.Label(ctrl, text="輝度しきい値", bg="#2b2b2b", fg="#aaaaaa",
                 font=("", 9)).pack(anchor=tk.W, padx=12, pady=(4, 0))
        self.line_threshold_var = tk.IntVar(value=self.line_threshold)
        tk.Scale(
            ctrl, from_=0, to=255, orient=tk.HORIZONTAL,
            variable=self.line_threshold_var,
            command=lambda v: setattr(self, "line_threshold", int(v)),
            bg="#2b2b2b", fg="#cccccc", highlightthickness=0,
            troughcolor="#444", activebackground="#777", length=136,
        ).pack(padx=12)

        tk.Label(ctrl, text="膨張", bg="#2b2b2b", fg="#aaaaaa",
                 font=("", 9)).pack(anchor=tk.W, padx=12, pady=(6, 0))
        self.line_dilate_var = tk.IntVar(value=self.line_dilate_scale)
        tk.Scale(
            ctrl, from_=0, to=200, orient=tk.HORIZONTAL,
            variable=self.line_dilate_var,
            command=lambda v: setattr(self, "line_dilate_scale", int(v)),
            bg="#2b2b2b", fg="#cccccc", highlightthickness=0,
            troughcolor="#444", activebackground="#777", length=136,
        ).pack(padx=12)

        tk.Frame(ctrl, height=1, bg="#444").pack(fill=tk.X, padx=10, pady=(10, 4))

        tk.Button(
            ctrl, text="Undo  (Ctrl+Z)", command=self._undo,
            bg="#555555", fg="white", relief=tk.FLAT,
            activebackground="#777", bd=0, padx=8, pady=6,
        ).pack(fill=tk.X, padx=12, pady=4)

        # 右のサムネイル一覧。canvas より先に pack して右端を確保する
        side = tk.Frame(self.root, width=THUMB_PANEL_W, bg="#252525")
        side.pack(side=tk.RIGHT, fill=tk.Y)
        side.pack_propagate(False)

        self.thumb_canvas = tk.Canvas(side, bg="#252525", highlightthickness=0)
        thumb_sb = tk.Scrollbar(side, orient=tk.VERTICAL,
                                command=self.thumb_canvas.yview,
                                troughcolor="#252525", bg="#555", bd=0,
                                highlightthickness=0, takefocus=0)
        self.thumb_canvas.configure(yscrollcommand=thumb_sb.set)
        thumb_sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.thumb_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.thumb_canvas.bind("<Button-1>", self._on_thumb_click)
        self.thumb_canvas.bind("<MouseWheel>", self._on_thumb_scroll)

        self.canvas = tk.Canvas(self.root, bg="#111111", cursor="none",
                                highlightthickness=0)
        self.canvas.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.canvas.bind("<ButtonPress-1>", self._on_down)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_up)
        self.canvas.bind("<ButtonPress-3>", self._on_pan_start)
        self.canvas.bind("<B3-Motion>", self._on_pan_drag)
        self.canvas.bind("<ButtonRelease-3>", self._on_pan_end)
        self.canvas.bind("<MouseWheel>", self._on_scroll)
        self.canvas.bind("<Shift-MouseWheel>", self._on_shift_scroll)
        self.canvas.bind("<Motion>", self._on_motion)
        self.canvas.bind("<Leave>", self._on_leave)
        self.canvas.bind("<Configure>", self._on_canvas_resize)
        self.root.bind("<Control-z>", self._undo)
        self.root.bind("<Control-Z>", self._undo)
        for seq in ("<Left>", "<Up>"):
            self.root.bind(seq, self._on_key_prev)
        for seq in ("<Right>", "<Down>"):
            self.root.bind(seq, self._on_key_next)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Scale はフォーカスがあると矢印キーで値が動いてしまう（左右だけでなく上下も）。
        # ウィジェット単位のバインディングはクラスバインディングより先に走るので、
        # そこで break して潰す
        for w in self._descendants(ctrl):
            for seq in ("<Left>", "<Up>"):
                w.bind(seq, self._on_key_prev)
            for seq in ("<Right>", "<Down>"):
                w.bind(seq, self._on_key_next)

        self._set_tool("brush")

    def _descendants(self, widget):
        for child in widget.winfo_children():
            yield child
            yield from self._descendants(child)

    # ── ツール選択 ────────────────────────────────────────────

    def _set_brush(self, size):
        # 太さの変更は消しゴム・黒線ブラシ中にも効かせたいので、ツールは切り替えない
        # （ラジオボタンだった頃はブラシに切り替えていたが、スライダーだと
        #   ドラッグの途中で毎回ツールが戻ってしまうため）
        self.brush_size = size
        self._redraw_cursor()

    def _set_brush_shape(self, shape):
        # 形の変更は消しゴム・黒線ブラシ中にも効かせたいので、ツールは切り替えない
        self.brush_shape = shape
        self._redraw_cursor()

    def _set_tool(self, tool):
        self.tool = tool
        for t, btn in self.tool_buttons.items():
            btn.config(relief=tk.SUNKEN if t == tool else tk.RAISED,
                       bg="#888888" if t == tool else "#555555")
        self._redraw_cursor()

    # ── 画像管理 ──────────────────────────────────────────────

    def _build_entries(self):
        """input の未コピー分を output へ複製し、output の画像一覧を作る

        コピーは shutil.copy2（バイナリ）で行う。Pillow で開いて保存し直すと
        JPEG が再エンコードされて、一度も編集していない画像まで劣化するため。
        """
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        if os.path.isdir(INPUT_DIR):
            for name in sorted(os.listdir(INPUT_DIR)):
                if not name.lower().endswith(IMAGE_EXTS):
                    continue
                dst = os.path.join(OUTPUT_DIR, name)
                if os.path.exists(dst):
                    continue  # 既存の作業結果は上書きしない
                try:
                    shutil.copy2(os.path.join(INPUT_DIR, name), dst)
                except OSError as e:
                    self.copy_errors.append(f"{name}: {e.strerror or e}")

        names = sorted(f for f in os.listdir(OUTPUT_DIR)
                       if f.lower().endswith(IMAGE_EXTS))
        # 元画像は input にあればそちら。無ければ output のファイル自体が元になる
        # （過去のセッションで焼き込んだモザイクは剥がせない）
        self.entries = []
        for name in names:
            src = os.path.join(INPUT_DIR, name)
            self.entries.append(
                (name, src if os.path.exists(src) else os.path.join(OUTPUT_DIR, name))
            )

    def _select_index(self, index):
        """指定の画像へ切り替える。現在のマスクは保持し、必要なら保存する"""
        if not self.entries:
            return
        index = max(0, min(index, len(self.entries) - 1))
        if self.original_image is not None:
            # 保留中の保存は待たずにここで確定させる
            self._cancel_pending_save()
            self._save_current()
            self._store_mask()
        self.current_index = index
        self._load_current()
        self._update_thumb_selection()
        self._scroll_to_current()

    def _load_current(self):
        name, src = self.entries[self.current_index]
        try:
            self.original_image = Image.open(src).convert("RGB")
        except OSError as e:
            messagebox.showwarning("警告", f"{name} を開けなかったのだ:\n{e}")
            self.original_image = None
            self.composite_image = None
            self.mask_image = None
            self.canvas.delete("img")
            self._img_item = None
            self.status_var.set(f"{name} を開けなかったのだ")
            return
        iw, ih = self.original_image.size

        stored = self.masks.get(name)
        if stored is not None:
            mask = Image.open(io.BytesIO(stored)).convert("L")
            # 元画像が差し替わっていた場合に備えてサイズを確認する
            self.mask_image = mask if mask.size == (iw, ih) else Image.new("L", (iw, ih), 0)
        else:
            self.mask_image = Image.new("L", (iw, ih), 0)

        # undo_stack は全画像で1本なのでここではクリアしない
        self._img_item = None
        self.canvas.delete("img")

        block = max(MOSAIC_BLOCK_MIN, iw // MOSAIC_BLOCK_RATIO)
        bw, bh = max(1, iw // block), max(1, ih // block)
        self.pixelated_image = (
            self.original_image.resize((bw, bh), Image.BOX).resize((iw, ih), Image.NEAREST)
        )

        self.status_var.set(f"{self.current_index + 1} / {len(self.entries)}\n\n{name}")
        self.root.title(
            f"Mosaic Tool  [{self.current_index + 1}/{len(self.entries)}]  {name}"
        )

        self._fit_image()
        self._rebuild_composite()
        self._refresh_canvas()

    def _store_mask(self):
        """現在のマスクを PNG に圧縮して保持する。一度も塗っていなければ捨てる"""
        if self.mask_image is None:
            return
        name = self.entries[self.current_index][0]
        if self.mask_image.getbbox() is None:
            self.masks.pop(name, None)
            return
        buf = io.BytesIO()
        self.mask_image.save(buf, format="PNG")
        self.masks[name] = buf.getvalue()

    def _save_current(self):
        """output のファイルを更新する。一度も塗っていない画像は触らない

        起動時にコピーした状態のままなので、書き戻すと JPEG が無駄に劣化する。
        """
        if self.composite_image is None or self.mask_image is None:
            return
        name = self.entries[self.current_index][0]
        # まだ塗られていない画像は起動時のコピーのままなので触らない。ただし一度保存した
        # 後に Undo で全消しした場合は、塗る前の状態を書き戻す必要がある
        if self.mask_image.getbbox() is None and name not in self.saved:
            return
        path = os.path.join(OUTPUT_DIR, name)
        try:
            if os.path.splitext(name)[1].lower() in (".jpg", ".jpeg"):
                self.composite_image.save(path, quality=JPEG_QUALITY)
            else:
                self.composite_image.save(path)
            self.saved.add(name)
        except OSError as e:
            self.status_var.set(f"保存できなかったのだ\n{name}\n{e.strerror or e}")

    def _flush_current(self):
        """1操作終わるごとに output を更新し、一覧の印も更新する

        保存は少し遅らせて、続けて塗っている間は最後の1回にまとめる。1024x1024 の
        PNG 書き出しは 100ms 超えることがあり、ストロークのたびに走ると引っかかる。
        """
        self._store_mask()
        self._update_thumb_selection()
        self._cancel_pending_save()
        self._save_after_id = self.root.after(SAVE_DELAY_MS, self._save_now)

    def _save_now(self):
        self._save_after_id = None
        self._save_current()

    def _cancel_pending_save(self):
        if self._save_after_id is not None:
            self.root.after_cancel(self._save_after_id)
            self._save_after_id = None

    def _step(self, delta):
        # ドラッグ中に切り替えると、以降の _apply_point が切り替え先のマスクに
        # 書き込んでしまう（_push_undo は切り替え前の画像で積まれている）。
        # キーでも起きるが、Shift+ホイールは同じ手で操作できるぶん現実に起きる
        if self._drawing:
            return
        self._select_index(self.current_index + delta)

    def _on_key_prev(self, event=None):
        self._step(-1)
        return "break"

    def _on_key_next(self, event=None):
        self._step(1)
        return "break"

    def _on_close(self):
        # 閉じる直前の1枚も取りこぼさない（保留中の保存を確定させる）
        self._cancel_pending_save()
        if self.original_image is not None:
            self._save_current()
        self.root.destroy()

    def _fit_image(self):
        cw = max(self.canvas.winfo_width(), 1)
        ch = max(self.canvas.winfo_height(), 1)
        iw, ih = self.original_image.size
        self.scale = min(cw / iw, ch / ih, 1.0)
        dw, dh = int(iw * self.scale), int(ih * self.scale)
        self.offset_x = (cw - dw) // 2
        self.offset_y = (ch - dh) // 2

    # ── Undo ─────────────────────────────────────────────────

    def _push_undo(self):
        """操作前のマスクを、どの画像のものかと一緒に積む

        履歴は全画像で1本。マスクは PNG 圧縮して持つので、素の L 画像のように
        1段あたり画像サイズぶん（1024x1024 で 1MB）は食わない。
        """
        if self.mask_image is None:
            return
        buf = io.BytesIO()
        self.mask_image.save(buf, format="PNG")
        self.undo_stack.append((self.entries[self.current_index][0], buf.getvalue()))

    def _undo(self, event=None):
        """1操作戻す。別の画像の操作だったら、その画像へ移動してから戻す"""
        if not self.undo_stack:
            return "break"
        name, data = self.undo_stack.pop()
        index = next((i for i, (n, _) in enumerate(self.entries) if n == name), None)
        if index is None:
            return "break"  # 一覧から消えたファイル
        if index != self.current_index:
            # 移動する。現在の画像の状態はここで保存・保持される
            self._select_index(index)
        if self.original_image is None:
            return "break"
        mask = Image.open(io.BytesIO(data)).convert("L")
        if mask.size != self.original_image.size:
            return "break"
        self.mask_image = mask
        self._rebuild_composite()
        self._refresh_canvas()
        self._flush_current()
        return "break"

    # ── モザイク処理 ──────────────────────────────────────────

    def _rebuild_composite(self):
        self.composite_image = Image.composite(
            self.pixelated_image, self.original_image, self.mask_image
        )

    def _apply_mask_region(self, x1, y1, x2, y2):
        """composite_image の矩形領域だけ差分更新する"""
        iw, ih = self.original_image.size
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(iw, x2), min(ih, y2)
        if x2 <= x1 or y2 <= y1:
            return
        region = (x1, y1, x2, y2)
        merged = Image.composite(
            self.pixelated_image.crop(region),
            self.original_image.crop(region),
            self.mask_image.crop(region),
        )
        self.composite_image.paste(merged, (x1, y1))

    def _paint_brush(self, ix, iy, radius, erase):
        draw = ImageDraw.Draw(self.mask_image)
        color = 0 if erase else 255
        box = [ix - radius, iy - radius, ix + radius, iy + radius]
        if self.brush_shape == "square":
            draw.rectangle(box, fill=color)
        else:
            draw.ellipse(box, fill=color)
        self._apply_mask_region(
            int(ix - radius), int(iy - radius),
            int(ix + radius) + 1, int(iy + radius) + 1,
        )

    def _magic_wand(self, ix, iy):
        iw, ih = self.original_image.size
        ix, iy = int(round(ix)), int(round(iy))
        if not (0 <= ix < iw and 0 <= iy < ih):
            return

        img_arr = np.array(self.original_image)
        target = img_arr[iy, ix].astype(np.int32)

        # クリック点と色差が tolerance 以内のピクセルマスク
        diff = np.abs(img_arr.astype(np.int32) - target).max(axis=2)
        similar = diff <= self.wand_tolerance

        # BFS で連結成分を取得
        visited = np.zeros((ih, iw), dtype=bool)
        visited[iy, ix] = True
        queue = deque([(ix, iy)])
        xs, ys = [], []

        while queue:
            x, y = queue.popleft()
            xs.append(x)
            ys.append(y)
            for dx, dy in ((1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)):
                nx, ny = x + dx, y + dy
                if 0 <= nx < iw and 0 <= ny < ih and not visited[ny, nx] and similar[ny, nx]:
                    visited[ny, nx] = True
                    queue.append((nx, ny))

        if not xs:
            return

        # BFS 結果を一時マスクに書いて dilate → 境界線を飲み込む
        dilate_px = int(iw // 100 * self.wand_dilate_scale / 100)
        region_mask = np.zeros((ih, iw), dtype=np.uint8)
        region_mask[ys, xs] = 255
        if dilate_px > 0:
            kernel = dilate_px * 2 + 1
            dilated = Image.fromarray(region_mask, mode='L').filter(
                ImageFilter.MaxFilter(kernel)
            )
        else:
            dilated = Image.fromarray(region_mask, mode='L')

        mask_arr = np.array(self.mask_image)
        mask_arr = np.maximum(mask_arr, np.array(dilated))
        self.mask_image = Image.fromarray(mask_arr)

        pad = dilate_px + 1
        x1, x2 = max(0, min(xs) - pad), min(iw, max(xs) + pad + 1)
        y1, y2 = max(0, min(ys) - pad), min(ih, max(ys) + pad + 1)
        self._apply_mask_region(x1, y1, x2, y2)

    def _paint_line(self, ix, iy, radius):
        """ブラシ形状内の暗い（黒線）ピクセルだけを mask=255 にする"""
        iw, ih = self.original_image.size
        # 膨張分だけ bbox を広げて、膨張が切れないようにする
        dilate_px = int(iw // 100 * self.line_dilate_scale / 100)
        pad = dilate_px + 1
        x1 = max(0, int(ix - radius) - pad)
        y1 = max(0, int(iy - radius) - pad)
        x2 = min(iw, int(ix + radius) + pad + 1)
        y2 = min(ih, int(iy + radius) + pad + 1)
        if x2 <= x1 or y2 <= y1:
            return

        # bbox 内のブラシ形状
        yy, xx = np.ogrid[y1:y2, x1:x2]
        if self.brush_shape == "square":
            shape = (np.abs(xx - ix) <= radius) & (np.abs(yy - iy) <= radius)
        else:
            shape = (xx - ix) ** 2 + (yy - iy) ** 2 <= radius ** 2

        # 輝度で黒線を判定（しきい値以下を黒線とみなす）
        crop = np.array(self.original_image.crop((x1, y1, x2, y2))).astype(np.int32)
        lum = (crop[:, :, 0] * 299 + crop[:, :, 1] * 587 + crop[:, :, 2] * 114) // 1000
        dark = (lum <= self.line_threshold) & shape

        region = np.where(dark, 255, 0).astype(np.uint8)
        if dilate_px > 0:
            kernel = dilate_px * 2 + 1
            region = np.array(
                Image.fromarray(region, mode="L").filter(ImageFilter.MaxFilter(kernel))
            )

        # bbox のみ読み書きしてマスクへ合成
        sub = np.array(self.mask_image.crop((x1, y1, x2, y2)))
        merged = np.maximum(sub, region).astype(np.uint8)
        self.mask_image.paste(Image.fromarray(merged, mode="L"), (x1, y1))
        self._apply_mask_region(x1, y1, x2, y2)

    # ── 表示 ─────────────────────────────────────────────────

    def _refresh_canvas(self):
        if self.composite_image is None:
            return
        iw, ih = self.composite_image.size
        dw = max(1, int(iw * self.scale))
        dh = max(1, int(ih * self.scale))
        display = self.composite_image.resize((dw, dh), Image.NEAREST)
        self.tk_image = ImageTk.PhotoImage(display)
        if self._img_item is None:
            self._img_item = self.canvas.create_image(
                self.offset_x, self.offset_y, anchor=tk.NW,
                image=self.tk_image, tags="img",
            )
        else:
            self.canvas.itemconfig(self._img_item, image=self.tk_image)
            self.canvas.coords(self._img_item, self.offset_x, self.offset_y)
            if self.canvas.find_withtag("cursor"):
                self.canvas.tag_lower("img", "cursor")
        self._redraw_cursor()

    # ── カーソル ─────────────────────────────────────────────

    def _redraw_cursor(self):
        self.canvas.delete("cursor")
        if self._cursor_pos is None:
            return
        cx, cy = self._cursor_pos
        if self.tool == "wand":
            # 十字カーソル
            color = "#ffdd44"
            for dx, dy, ex, ey in [(-12,0,-3,0),(3,0,12,0),(0,-12,0,-3),(0,3,0,12)]:
                self.canvas.create_line(cx+dx, cy+dy, cx+ex, cy+ey,
                                        fill="#000", width=3, tags="cursor")
                self.canvas.create_line(cx+dx, cy+dy, cx+ex, cy+ey,
                                        fill=color, width=1, tags="cursor")
        else:
            r = self.brush_size
            color = {"eraser": "#ff6666", "line": "#44ddff"}.get(self.tool, "#ffffff")
            outline_shape = (self.canvas.create_rectangle
                             if self.brush_shape == "square"
                             else self.canvas.create_oval)
            outline_shape(cx-r, cy-r, cx+r, cy+r,
                          outline="#000000", width=3, tags="cursor")
            outline_shape(cx-r, cy-r, cx+r, cy+r,
                          outline=color, width=1, tags="cursor")
            # 中心の十字。細いブラシだと外周からはみ出すので、半径に合わせて縮める
            m = min(4, max(1, r - 2))
            self.canvas.create_line(cx-m, cy, cx+m, cy, fill=color, width=1, tags="cursor")
            self.canvas.create_line(cx, cy-m, cx, cy+m, fill=color, width=1, tags="cursor")

    def _on_motion(self, event):
        self._cursor_pos = (event.x, event.y)
        self._redraw_cursor()

    def _on_leave(self, event):
        self._cursor_pos = None
        self.canvas.delete("cursor")

    # ── ズーム ────────────────────────────────────────────────

    def _on_scroll(self, event):
        if self.original_image is None:
            return
        factor = ZOOM_FACTOR if event.delta > 0 else 1.0 / ZOOM_FACTOR
        new_scale = max(ZOOM_MIN, min(self.scale * factor, ZOOM_MAX))
        mx, my = event.x, event.y
        ix = (mx - self.offset_x) / self.scale
        iy = (my - self.offset_y) / self.scale
        self.scale = new_scale
        self.offset_x = int(mx - ix * new_scale)
        self.offset_y = int(my - iy * new_scale)
        self._refresh_canvas()

    def _on_shift_scroll(self, event):
        # Shift + ホイールで画像を切り替える（上で前、下で次。↑↓キーと同じ）
        if event.delta == 0:
            return "break"
        self._step(-1 if event.delta > 0 else 1)
        return "break"

    # ── パン ─────────────────────────────────────────────────

    def _on_pan_start(self, event):
        self._pan_start = (event.x, event.y)
        self._pan_offset_start = (self.offset_x, self.offset_y)
        self.canvas.config(cursor="fleur")

    def _on_pan_drag(self, event):
        if self._pan_start is None:
            return
        dx = event.x - self._pan_start[0]
        dy = event.y - self._pan_start[1]
        self.offset_x = self._pan_offset_start[0] + dx
        self.offset_y = self._pan_offset_start[1] + dy
        self._refresh_canvas()

    def _on_pan_end(self, event):
        self._pan_start = None
        self.canvas.config(cursor="none")

    # ── マウス操作（描画） ────────────────────────────────────

    def _canvas_to_image(self, cx, cy):
        return (cx - self.offset_x) / self.scale, (cy - self.offset_y) / self.scale

    def _on_down(self, event):
        if self.original_image is None:
            return
        if self.tool == "wand":
            self._push_undo()
            ix, iy = self._canvas_to_image(event.x, event.y)
            self._magic_wand(ix, iy)
            self._refresh_canvas()
            self._flush_current()
        else:
            self._push_undo()
            self._drawing = True
            self._apply_point(event.x, event.y)

    def _on_drag(self, event):
        if self._drawing:
            self._apply_point(event.x, event.y)
        self._cursor_pos = (event.x, event.y)

    def _on_up(self, event):
        # 保存はストロークの終わりに1回だけ。ドラッグ中に毎回書くと重い
        if self._drawing:
            self._drawing = False
            self._flush_current()

    def _apply_point(self, cx, cy):
        ix, iy = self._canvas_to_image(cx, cy)
        radius = self.brush_size / self.scale
        if self.tool == "line":
            self._paint_line(ix, iy, radius)
        else:
            self._paint_brush(ix, iy, radius, erase=(self.tool == "eraser"))
        self._refresh_canvas()

    # ── サムネイル一覧 ────────────────────────────────────────

    def _build_thumb_cells(self):
        """先に空のセルを並べておき、サムネイルは出来次第あとから流し込む"""
        cw = THUMB_PANEL_W - 20
        for i, (name, _) in enumerate(self.entries):
            y = i * THUMB_CELL_H
            self.thumb_canvas.create_rectangle(
                4, y + 4, cw + 4, y + THUMB_CELL_H - 4,
                fill="#303030", outline="", tags=("cell", f"cell{i}"),
            )
            self.thumb_items[i] = self.thumb_canvas.create_image(
                (cw + 8) // 2, y + THUMB_CELL_H // 2, tags=("thumb", f"thumb{i}"),
            )
            # 塗り済みの印（マスクを持っている画像だけ表示する）
            self.thumb_marks[i] = self.thumb_canvas.create_oval(
                cw - 6, y + 10, cw + 2, y + 18,
                fill="#4ec98a", outline="", state=tk.HIDDEN, tags=f"mark{i}",
            )
        self.thumb_canvas.configure(
            scrollregion=(0, 0, cw + 8, max(1, len(self.entries)) * THUMB_CELL_H)
        )

    def _start_thumb_loader(self):
        """サムネイル生成はバックグラウンドで行う（枚数が多いと起動が固まるため）

        PIL の処理だけスレッドで行い、出来たものはキューに積む。tkinter は他スレッドから
        触れないうえ、root.after すら別スレッドから呼ぶと mainloop 開始前に
        RuntimeError になるので、取り出しはメインスレッドのポーリングに任せる。
        """
        threading.Thread(target=self._thumb_worker, daemon=True).start()
        self.root.after(50, self._drain_thumb_queue)

    def _thumb_worker(self):
        for i, (name, _) in enumerate(self.entries):
            try:
                im = Image.open(os.path.join(OUTPUT_DIR, name))
                im.draft("RGB", (THUMB_W * 2, THUMB_W * 2))  # JPEG は間引いて読む
                im = im.convert("RGB")
                im.thumbnail((THUMB_W, THUMB_W - 12), Image.BILINEAR)
            except (OSError, ValueError):
                continue
            self._thumb_queue.put((i, im))
        self._thumb_done = True

    def _drain_thumb_queue(self):
        try:
            while True:
                index, im = self._thumb_queue.get_nowait()
                self._set_thumb(index, im)
        except queue.Empty:
            pass
        if not self._thumb_done or not self._thumb_queue.empty():
            self.root.after(50, self._drain_thumb_queue)

    def _set_thumb(self, index, im):
        if index not in self.thumb_items:
            return
        try:
            photo = ImageTk.PhotoImage(im)
        except tk.TclError:
            return  # ウィンドウが閉じられた後
        self.thumb_photos[index] = photo  # 参照を保持しないと GC されて消える
        self.thumb_canvas.itemconfig(self.thumb_items[index], image=photo)

    def _update_thumb_selection(self):
        cw = THUMB_PANEL_W - 20
        y = self.current_index * THUMB_CELL_H
        if self.thumb_sel is None:
            self.thumb_sel = self.thumb_canvas.create_rectangle(
                4, y + 4, cw + 4, y + THUMB_CELL_H - 4,
                outline="#4ec98a", width=2, tags="sel",
            )
        else:
            self.thumb_canvas.coords(
                self.thumb_sel, 4, y + 4, cw + 4, y + THUMB_CELL_H - 4
            )
        self.thumb_canvas.tag_raise("sel")
        for i, (name, _) in enumerate(self.entries):
            self.thumb_canvas.itemconfig(
                self.thumb_marks[i],
                state=tk.NORMAL if name in self.masks else tk.HIDDEN,
            )

    def _scroll_to_current(self):
        total = max(1, len(self.entries)) * THUMB_CELL_H
        view_h = max(1, self.thumb_canvas.winfo_height())
        if total <= view_h:
            return
        top = self.thumb_canvas.canvasy(0)
        y = self.current_index * THUMB_CELL_H
        if y < top:
            self.thumb_canvas.yview_moveto(y / total)
        elif y + THUMB_CELL_H > top + view_h:
            self.thumb_canvas.yview_moveto((y + THUMB_CELL_H - view_h) / total)

    def _on_thumb_click(self, event):
        index = int(self.thumb_canvas.canvasy(event.y) // THUMB_CELL_H)
        if 0 <= index < len(self.entries) and index != self.current_index:
            self._select_index(index)

    def _on_thumb_scroll(self, event):
        self.thumb_canvas.yview_scroll(-1 if event.delta > 0 else 1, "units")

    # ── ウィンドウリサイズ ────────────────────────────────────

    def _on_canvas_resize(self, event):
        if self.original_image is not None:
            self._fit_image()
            self._refresh_canvas()


def main():
    root = tk.Tk()
    app = MosaicApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
