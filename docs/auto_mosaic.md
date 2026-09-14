# 自動モザイク（auto_mosaic.py）

画像の中の局部を AI で見つけて、その範囲にだけ自動でモザイクをかける仕組みなのだ。
**手作業の置き換えではなく、下塗り**として使う。自動でかけたあとは、いつもの `run.bat` で1枚ずつ確認して、漏れをブラシで塗り足す。

- コマンドで使う: `auto_mosaic.py`（フォルダの画像をまとめて処理）
- ComfyUI で使う: `comfyui/mozaic_auto`（生成した画像を保存する直前にかける）

どちらも中身は同じ `auto_mosaic.py` の関数なのだ。

---

## 仕組み

```
元画像 ─┬─→ ① 検出 ─→ ② 範囲を広げる ─→ マスク ─┐
        │                                          ├─→ ④ 合成 ─→ 結果
        └─→ ③ 画像全体をモザイクにする ────────────┘
```

| 段 | やること | 既定値 |
|---|---|---|
| ① 検出 | YOLO のセグメンテーションモデルで局部を探し、**形のとおり**に白く塗ったマスクを作る。モデルを複数指定すると結果を足し合わせる | モデル1本・信頼度 0.35 以上 |
| ② 範囲を広げる | マスクを外側へ太らせる。輪郭ぴったりだと縁が残るため | 30px |
| ③ モザイク画像 | 画像全体を縮小（BOX）→ 拡大（NEAREST）する。**粒度は app.py と同じ**（画像幅の 1/100・最小 4px） | — |
| ④ 合成 | マスクが白いところだけ ③ の画像に差し替える | — |

**検出が1件も無かった画像には何もしない**（元のファイルをそのままコピーする）。

### セグメンテーションと枠（bbox）
YOLO のモデルには2種類ある。

- **セグメンテーション（segm）**: 部位の形をピクセル単位で返す。モザイクが形に沿うので、かかりすぎない
- **枠（bbox / detect）**: 部位を囲む四角だけを返す。この場合は四角ごと塗る

どちらのモデルを指定しても動くのだ。

---

## セットアップ

### 1. いつもの setup.bat を済ませる
`venv/` ができていればよい。

### 2. 追加のパッケージを入れる

```
venv\Scripts\pip install -r requirements-auto.txt
```

`ultralytics`（YOLO の実行ライブラリ）が入る。PyTorch も一緒に入るので、少し時間がかかる。GPU が無くても CPU で動くのだ。

### 3. 検出モデルを置く
既定は **Anime NSFW segm/detailer** の `xl-res1280` 版。

1. https://civitai.com/models/2619511 を開く（成人向けモデルなので、Civitai へのログインと表示設定が要る）
2. バージョン **xl-res1280** の `animeNSFWSegm_xlRes1280.pt`（約135MB）をダウンロードする
3. `models/segm/` に置く

```
mosaic/
└── models/
    └── segm/
        └── animeNSFWSegm_xlRes1280.pt
```

壊れていないかは SHA256 で確かめられる。

```
certutil -hashfile models\segm\animeNSFWSegm_xlRes1280.pt SHA256
```

`add48b80c24755c06a17978610fcce20203bd40d4cc6b503d2319a2ccb3f41e7` と一致すればよい（大文字小文字は問わない）。

> ⚠️ `.pt` ファイルは Python の pickle 形式で、**読み込むと中のコードが実行されうる**。配布元が確かなものだけを使うのだ。

---

## 使い方（コマンド）

### A. 全部を GUI で確認する（おすすめ）

```
venv\Scripts\python auto_mosaic.py D:\generated
run.bat
```

1. `D:\generated` の画像をすべて処理して `input/` へ書き出す
   - 検出できた画像 → モザイクをかけて保存
   - 検出ゼロの画像 → そのままコピー
2. `run.bat` で1枚ずつ見て、足りないところをブラシで塗る

最後に「検出ゼロ」の画像名が一覧で出るので、そこから先に見るとよい。

### B. 検出ゼロの画像だけを手作業にまわす

```
venv\Scripts\python auto_mosaic.py D:\generated --out D:\done --misses input
```

- 検出できた画像 → モザイクをかけて `D:\done` へ
- 検出ゼロの画像 → そのまま `input/` へ（`run.bat` で塗る）

`D:\done` の画像に塗り足したくなったら、そのファイルを `input/` へ移してから `run.bat` を起動するのだ。

### なぜ output/ ではなく input/ に書き出すのか
app.py は塗るたびに **`input/` の画像を元にして** `output/` を作り直す。
自動モザイクの結果を `output/` にだけ置いておくと、GUI で1筆塗った瞬間に `input/` の元画像から作り直されて、**自動でかけたモザイクが消える**。
`input/` に置けば、それが元画像になるので、GUI の塗りは自動モザイクの上に重なるのだ。

### 数えるだけ（書き出さない）

```
venv\Scripts\python auto_mosaic.py D:\generated --measure
```

モデルや信頼度を変えたときに、何枚で検出できるかを先に確かめるのに使う。

### オプション

| オプション | 既定 | 内容 |
|---|---|---|
| `src` | （必須） | 元画像のフォルダ。**直下の画像だけ**を処理する（app.py と同じ） |
| `--out` | `input/` | モザイクをかけた画像の書き出し先 |
| `--misses` | `--out` と同じ | 検出ゼロの画像のコピー先 |
| `--models` | `models/segm/animeNSFWSegm_xlRes1280.pt` | モデルのパス。カンマ区切りで4本まで。相対パスは auto_mosaic.py のあるフォルダから |
| `--conf` | 0.35 | 信頼度の下限。下げると拾う数が増え、誤検出も増える |
| `--expand` | 30 | マスクを外へ広げる幅（px） |
| `--grain` | 0 | モザイクの1マス（px）。0 は app.py と同じ粒度 |
| `--classes` | `penis,vagina,anus,pussy,genitalia_f` | 隠すクラス名。モデルに無い名前は無視される |
| `--device` | `cpu` | `0` にすると1枚目の GPU を使う |
| `--overwrite` | なし | 書き出し先に同名ファイルがあっても処理し直す |
| `--measure` | なし | 検出して数えるだけ |
| `--report` | なし | 検出の一覧（ファイル名・クラス・信頼度）を JSON で保存する |

### 安全のための動き
- **元画像のフォルダは書き換えない**。`--out` や `--misses` に元画像と同じフォルダを指定するとエラーで止まる
- **書き出し先に同名ファイルがあればスキップする**。GUI で塗り足した画像を、もう一度実行して潰さないため。処理し直すときだけ `--overwrite` を付ける
- 検出ゼロの画像はバイナリコピーなので、JPEG でも劣化しない。モザイクをかけた画像は保存し直すので、app.py と同じくメタデータ（EXIF・生成パラメータ）が落ち、JPEG は品質 95 で再エンコードされる

---

## ComfyUI で使う

生成した画像を、保存する前に自動でモザイクにするノードなのだ。

```
VAE Decode ─→ Auto Mosaic (YOLO SEGM) ─→ Save Image
                        └─ mask（どこにかけたかの白黒。使わなくてよい）
```

### 1. ComfyUI の Python に ultralytics を入れる
ComfyUI-Impact-Pack / Subpack を使っていれば入っている。無ければ ComfyUI の Python で入れる。

```
C:\path\to\ComfyUI\python_embeded\python.exe -m pip install ultralytics
```

（venv で動かしている場合は、その venv の python.exe を使う）

### 2. ノードを custom_nodes に置く
**リンクを張る方法（おすすめ）**。リポジトリを更新するとノードも一緒に更新される。

```
mklink /J "C:\path\to\ComfyUI\custom_nodes\mozaic_auto" "C:\path\to\mosaic\comfyui\mozaic_auto"
```

**コピーする方法**。`comfyui/mozaic_auto` フォルダを `custom_nodes` へコピーし、その中へ `auto_mosaic.py` もコピーする。

```
ComfyUI/custom_nodes/mozaic_auto/
├── __init__.py
└── auto_mosaic.py
```

### 3. モデルを置いて ComfyUI を再起動する
モデルは `ComfyUI/models/ultralytics/segm/`（枠のモデルなら `bbox/`）に置く。Impact-Pack と同じ置き場所なので、既に置いてあればそのまま使える。

再起動すると、ノード追加メニューの **mozaic → Auto Mosaic (YOLO SEGM)** に出る。

| 設定 | 内容 |
|---|---|
| model_1〜4 | 使うモデル。2〜4 は `none` で使わない |
| confidence | 信頼度の下限（既定 0.35） |
| coverage_expansion | マスクを外へ広げる幅 px（既定 30） |
| grain_size | モザイクの1マス px（0 = 画像幅の 1/100） |
| classes | 隠すクラス名（カンマ区切り） |

GPU があれば GPU で検出する。ComfyUI のコンソールに `[mozaic auto] 検出 N 件` と出るので、効いているかはそこで分かるのだ。

---

## どれくらい見つかるか（実測）

2026-09-14 に、SDXL で生成したケモノ系（獣人）の成人向け画像で測った。ultralytics 8.4.146・CPU。

### モデルの比較（性行為の画像 67 枚）
局部を1件以上検出できた枚数。

| モデル | 種類 | 検出できた枚数 | クラス |
|---|---|---|---|
| [Anime NSFW segm/detailer](https://civitai.com/models/2619511) xl-res1280（**既定**） | segm | 陰茎 42・女性器 32・肛門 8（どれか1つ以上: **56 枚 = 84%**） | anus, nipple, penis, vagina, female face, male face, pubic hair |
| [ADetailer - Anime pussy detection](https://civitai.com/models/2795759) | bbox | 20 | pussy |
| [Pussy + Pubes + Anus Detector](https://civitai.com/models/1659913) | segm | 19 | pussy |
| [YOLOv8m-seg Female Genitalia Specialist v2](https://civitai.com/models/2720283) | segm | 12 | genitalia_f |
| [YOLO Penis Detection](https://civitai.com/models/2229108) | bbox | 0 | penis |

1本目が他を大きく上回ったので、既定は1本だけにしてある。

### 場面ごと（既定モデル・1場面 30 枚）

| 場面 | 検出できた枚数 |
|---|---|
| 前戯・フェラ・正常位・後背位・騎乗位 | 各 24〜28 |
| 絶頂 | 18 |
| 事後 | 11 |
| 着衣（露出なし） | 0（隠すものが無いので正しい） |

### 分かっている限界
- **見つからないのは、引きで小さく写った構図と、体で隠れた構図**が多い
- **誤検出もある**。机の上のランプにモザイクがかかった例があった
- 上の数は「1件以上見つかった枚数」で、**写っている部位の全部にかかったかは数えていない**。だから GUI での確認は省けないのだ
- 人間やアニメ絵では測っていない

---

## ライセンスまわり
- `ultralytics` は **AGPL-3.0**
- 検出モデルは Civitai の各ページに利用条件がある。使う前に確認するのだ

---

## ファイル

```
mosaic/
├── auto_mosaic.py            # 検出とモザイクの本体（コマンドとしても使う）
├── requirements-auto.txt     # 追加で入れるパッケージ（ultralytics）
├── models/segm/              # 検出モデルを置く（.pt は git 管理外）
├── comfyui/mozaic_auto/      # ComfyUI のノード（本体の関数を呼ぶだけ）
└── docs/auto_mosaic.md       # このファイル
```
