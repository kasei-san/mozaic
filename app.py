import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk, ImageDraw, ImageFilter
import numpy as np
from collections import deque
import os
import json

INPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "input")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
SETTINGS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "settings.json")
IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')
MOSAIC_BLOCK_RATIO = 100
MOSAIC_BLOCK_MIN = 4
# 高さは全ウィジェットが収まる実測値（長いファイル名でステータスが折り返した状態で 841）
# に余裕を持たせた値。足りなくなると Undo ボタンから先に隠れる
WINDOW_W, WINDOW_H = 1280, 880
WINDOW_SCREEN_MARGIN_W, WINDOW_SCREEN_MARGIN_H = 80, 100
BRUSH_SIZE_MIN, BRUSH_SIZE_MAX = 10, 50
BRUSH_SHAPES = [("○", "circle"), ("□", "square")]
ZOOM_FACTOR = 1.15
ZOOM_MIN = 0.05
ZOOM_MAX = 20.0

_DEFAULT_SETTINGS = {
    "wand_tolerance": 25,
    "wand_dilate_scale": 25,
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

        os.makedirs(OUTPUT_DIR, exist_ok=True)

        files = sorted(f for f in os.listdir(INPUT_DIR) if f.lower().endswith(IMAGE_EXTS))
        self.images = [os.path.join(INPUT_DIR, f) for f in files]
        self.current_index = 0

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

        # Undo: マスクのスナップショットスタック
        self.undo_stack = []

        self._setup_ui()

        if not self.images:
            messagebox.showwarning("警告", "input フォルダに画像がないのだ")
            root.destroy()
            return

        self.load_image()

    # ── UI構築 ──────────────────────────────────────────────

    def _setup_ui(self):
        ctrl = tk.Frame(self.root, width=160, bg="#2b2b2b")
        ctrl.pack(side=tk.LEFT, fill=tk.Y)
        ctrl.pack_propagate(False)

        # 完了ボタンとステータスは下端固定。pack は宣言順に領域を切り出すので、
        # ウィンドウが低いときでも確実に表示されるよう他より先に pack する
        # （side=BOTTOM を指定しても、後から pack すると領域が残っておらず消える）
        self.status_var = tk.StringVar()
        tk.Label(ctrl, textvariable=self.status_var,
                 bg="#2b2b2b", fg="#888888",
                 wraplength=148, justify=tk.CENTER).pack(side=tk.BOTTOM, pady=16, padx=6)

        tk.Button(
            ctrl, text="完了 →", command=self._complete,
            bg="#3a9f6e", fg="white", relief=tk.FLAT,
            activebackground="#4ec98a", bd=0,
            font=("", 12, "bold"), pady=10,
        ).pack(side=tk.BOTTOM, fill=tk.X, padx=12, pady=4)

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
            ctrl, from_=1, to=100, orient=tk.HORIZONTAL,
            variable=self.tolerance_var,
            command=lambda v: setattr(self, "wand_tolerance", int(v)),
            bg="#2b2b2b", fg="#cccccc", highlightthickness=0,
            troughcolor="#444", activebackground="#777", length=136,
        ).pack(padx=12)

        tk.Label(ctrl, text="境界膨張", bg="#2b2b2b", fg="#aaaaaa",
                 font=("", 9)).pack(anchor=tk.W, padx=12, pady=(6, 0))
        self.dilate_var = tk.IntVar(value=self.wand_dilate_scale)
        tk.Scale(
            ctrl, from_=0, to=200, orient=tk.HORIZONTAL,
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
        self.canvas.bind("<Motion>", self._on_motion)
        self.canvas.bind("<Leave>", self._on_leave)
        self.canvas.bind("<Configure>", self._on_canvas_resize)
        self.root.bind("<Control-z>", lambda e: self._undo())
        self.root.bind("<Control-Z>", lambda e: self._undo())

        self._set_tool("brush")

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

    def load_image(self):
        while self.current_index < len(self.images):
            fname = os.path.basename(self.images[self.current_index])
            if not os.path.exists(os.path.join(OUTPUT_DIR, fname)):
                break
            self.current_index += 1

        if self.current_index >= len(self.images):
            messagebox.showinfo("完了", "全ての画像を処理したのだ！")
            self.root.destroy()
            return

        path = self.images[self.current_index]
        self.original_image = Image.open(path).convert("RGB")
        iw, ih = self.original_image.size
        self.mask_image = Image.new("L", (iw, ih), 0)
        self.undo_stack = []
        self._img_item = None

        block = max(MOSAIC_BLOCK_MIN, iw // MOSAIC_BLOCK_RATIO)
        bw, bh = max(1, iw // block), max(1, ih // block)
        self.pixelated_image = (
            self.original_image.resize((bw, bh), Image.BOX).resize((iw, ih), Image.NEAREST)
        )

        fname = os.path.basename(path)
        self.status_var.set(f"{self.current_index + 1} / {len(self.images)}\n\n{fname}")
        self.root.title(f"Mosaic Tool  [{self.current_index + 1}/{len(self.images)}]  {fname}")

        self._fit_image()
        self._rebuild_composite()
        self._refresh_canvas()

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
        self.undo_stack.append(self.mask_image.copy())

    def _undo(self):
        if self.undo_stack:
            self.mask_image = self.undo_stack.pop()
            self._rebuild_composite()
            self._refresh_canvas()

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
            self.canvas.create_line(cx-4, cy, cx+4, cy, fill=color, width=1, tags="cursor")
            self.canvas.create_line(cx, cy-4, cx, cy+4, fill=color, width=1, tags="cursor")

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
        if self.tool == "wand":
            self._push_undo()
            ix, iy = self._canvas_to_image(event.x, event.y)
            self._magic_wand(ix, iy)
            self._refresh_canvas()
        else:
            self._push_undo()
            self._drawing = True
            self._apply_point(event.x, event.y)

    def _on_drag(self, event):
        if self._drawing:
            self._apply_point(event.x, event.y)
        self._cursor_pos = (event.x, event.y)

    def _on_up(self, event):
        self._drawing = False

    def _apply_point(self, cx, cy):
        ix, iy = self._canvas_to_image(cx, cy)
        radius = self.brush_size / self.scale
        if self.tool == "line":
            self._paint_line(ix, iy, radius)
        else:
            self._paint_brush(ix, iy, radius, erase=(self.tool == "eraser"))
        self._refresh_canvas()

    # ── 完了 ─────────────────────────────────────────────────

    def _complete(self):
        if self.original_image is None:
            return
        src_path = self.images[self.current_index]
        fname = os.path.basename(src_path)
        self.composite_image.save(os.path.join(OUTPUT_DIR, fname))
        self.current_index += 1
        self.load_image()

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
