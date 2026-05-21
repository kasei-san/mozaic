import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk, ImageDraw, ImageFilter
import numpy as np
from collections import deque
import os

INPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "input")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')
MOSAIC_BLOCK_RATIO = 100
MOSAIC_BLOCK_MIN = 4
BRUSH_SIZES = [10, 20, 30, 50, 80]
WAND_TOLERANCE = 50
ZOOM_FACTOR = 1.15
ZOOM_MIN = 0.05
ZOOM_MAX = 20.0


class MosaicApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Mosaic Tool")
        self.root.geometry("1280x820")
        self.root.configure(bg="#1e1e1e")

        os.makedirs(OUTPUT_DIR, exist_ok=True)

        files = sorted(f for f in os.listdir(INPUT_DIR) if f.lower().endswith(IMAGE_EXTS))
        self.images = [os.path.join(INPUT_DIR, f) for f in files]
        self.current_index = 0

        # tool: "brush" | "eraser" | "wand"
        self.tool = "brush"
        self.brush_size = 30

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

        self.wand_tolerance = WAND_TOLERANCE
        self.wand_dilate_scale = 100  # 100 = iw//100、0 = 膨張なし

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

        tk.Label(ctrl, text="筆の太さ", bg="#2b2b2b", fg="#cccccc",
                 font=("", 10, "bold")).pack(pady=(24, 6))

        self.brush_var = tk.IntVar(value=self.brush_size)
        for size in BRUSH_SIZES:
            rb = tk.Radiobutton(
                ctrl, text=f"  {size} px",
                variable=self.brush_var, value=size,
                bg="#2b2b2b", fg="#cccccc",
                selectcolor="#444444", activebackground="#2b2b2b",
                command=lambda s=size: self._set_brush(s),
            )
            rb.pack(anchor=tk.W, padx=16, pady=2)

        tk.Frame(ctrl, height=1, bg="#444").pack(fill=tk.X, padx=10, pady=14)

        self.tool_buttons = {}
        for label, tool in [("ブラシ", "brush"), ("消しゴム", "eraser"), ("魔法の杖", "wand")]:
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

        tk.Button(
            ctrl, text="Undo  (Ctrl+Z)", command=self._undo,
            bg="#555555", fg="white", relief=tk.FLAT,
            activebackground="#777", bd=0, padx=8, pady=6,
        ).pack(fill=tk.X, padx=12, pady=4)

        tk.Frame(ctrl, height=1, bg="#444").pack(fill=tk.X, padx=10, pady=14)

        tk.Button(
            ctrl, text="完了 →", command=self._complete,
            bg="#3a9f6e", fg="white", relief=tk.FLAT,
            activebackground="#4ec98a", bd=0,
            font=("", 12, "bold"), pady=10,
        ).pack(fill=tk.X, padx=12, pady=4)

        self.status_var = tk.StringVar()
        tk.Label(ctrl, textvariable=self.status_var,
                 bg="#2b2b2b", fg="#888888",
                 wraplength=148, justify=tk.CENTER).pack(pady=16, padx=6)

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

    # ── ツール選択 ────────────────────────────────────────────

    def _set_brush(self, size):
        self.brush_size = size
        self._set_tool("brush")

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
        draw.ellipse([ix - radius, iy - radius, ix + radius, iy + radius], fill=color)
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
            color = "#ff6666" if self.tool == "eraser" else "#ffffff"
            self.canvas.create_oval(cx-r, cy-r, cx+r, cy+r,
                                    outline="#000000", width=3, tags="cursor")
            self.canvas.create_oval(cx-r, cy-r, cx+r, cy+r,
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
