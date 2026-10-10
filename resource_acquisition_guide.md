# PMG-Nav 所需仓库、数据集、模型与材料获取手册

更新日期：2026-07-22  
执行状态：**本文件只记录获取方法；当前没有执行下载或 clone。**  
适用环境：中国境内 AutoDL Linux 服务器  
主 baseline：TIC-VLA  
本项目核心：Physical Memory Graph（PMG）

本文件是后续连接 AutoDL 后获取资源的唯一执行入口。获取顺序固定为：

1. 先检查磁盘、GPU 和网络；
2. 优先使用官方源；
3. 官方源失败后，按本文件指定的国内镜像或可信代理回退；
4. 下载完成后记录版本、commit 和校验信息；
5. 先取最小资源跑通推理，禁止一开始下载 538 GB 全量数据。

---

## 1. 资源结论与下载优先级

### 1.1 真正必须的最小集合

| 阶段 | 资源 | 是否必需 | 用途 | 推荐落盘目录 |
|---|---|---:|---|---|
| Stage 1A | TIC-VLA repo | 必需 | baseline 模型、open-loop、DynaNav、RL 入口 | `$PMG_ROOT/src/TIC-VLA` |
| Stage 1A | InternVL3-1B | 必需 | TIC-VLA 的 VLM backbone | `$PMG_ROOT/models/InternVL3-1B` |
| Stage 1A | TIC-VLA pretrained checkpoint | 必需 | 不从头 SFT，直接复现 baseline | `$PMG_ROOT/checkpoints/TIC-VLA` |
| Stage 1A | DynaNav JSON + data 子集 | 必需 | open-loop released evaluation | `$PMG_ROOT/data/TIC-VLA/DynaNav` |
| Stage 1B | Isaac Sim 5.0.0 Linux | 必需 | DynaNav closed-loop 仿真和相机渲染 | `$PMG_ROOT/isaacsim` |
| Stage 1B | DynaNav scene assets | 必需，已随 repo 发布 | closed-loop 场景；随 TIC-VLA repo 获取 | `$PMG_ROOT/src/TIC-VLA/DynaNav` |
| Stage 2 | GroundingDINO + Swin-T weights | 条件必需 | PMG 开放词汇 object anchor；annotation gate 后决定 | `$PMG_ROOT/src/GroundingDINO` |
| Stage 2 | SAM 2.1 tiny/small | 可选 | box 到 mask、跨帧 object tracking；MVP 可先不用 | `$PMG_ROOT/src/sam2` |
| Stage 3 | IsaacLab v2.2.1 | 必需 | TIC-VLA PPO 与 PMG tool policy RL | `$PMG_ROOT/src/IsaacLab` |
| Full SFT | TIC-VLA 全量数据 | 非首轮必需 | 重做完整 VLM/action SFT | `$PMG_ROOT/data/TIC-VLA` |

### 1.2 明确不要提前下载的内容

- TIC-VLA Hugging Face 数据仓库全量约 **538 GB**。首轮只下载 checkpoint 和 DynaNav 子集。
- SCAND、GND 原始仓库不单独下载；released TIC-VLA 数据已经提供处理后的对应部分。
- 不下载 InternVL3 更大的 2B/8B/14B 版本；baseline 固定使用 `InternVL3-1B`。
- 不下载完整 SA-V、COCO、LVIS；它们不是 PMG-Nav baseline 或 PMG MVP 的训练数据。
- 不提前下载 Habitat/HM3D；它只在 DynaNav annotation gate 失败时成为 fallback。
- 不从头进行 TIC-VLA 的 8×L40S full-parameter SFT；先使用发布 checkpoint。

### 1.3 不需要单独获取的上游项目

- Urban-Sim：TIC-VLA RL README 对其进行 acknowledgement，但当前 TIC-VLA repo 已发布所需环境实现和 DynaNav local assets；除非实际运行出现明确的缺失 import/asset path，不单独 clone。
- SCAND/GND 原始数据：TIC-VLA full dataset 中已有项目所用的处理版本，复现 TIC-VLA 不需要另取原始包。
- COCO/LVIS/SA-V：GroundingDINO/SAM2 使用发布权重做 inference，不在本项目中重新预训练 detector/segmenter。
- GOAT-Bench/HM3D-OVON/SafeVLA：属于辅助对照或 alternative，不是主 baseline 的环境依赖，不进入 Stage 1 资源清单。

---

## 2. AutoDL 统一目录与环境变量

后续在服务器上先定义一次：

```bash
export PMG_ROOT=/root/autodl-tmp/ccfb-ticvla
export PMG_SRC_ROOT="$PMG_ROOT/src"
export PMG_DATA_ROOT="$PMG_ROOT/data"
export PMG_MODEL_ROOT="$PMG_ROOT/models"
export PMG_CKPT_ROOT="$PMG_ROOT/checkpoints"
export PMG_DOWNLOAD_ROOT="$PMG_ROOT/downloads"
export PMG_MANIFEST_ROOT="$PMG_ROOT/manifests"

mkdir -p \
  "$PMG_SRC_ROOT" \
  "$PMG_DATA_ROOT" \
  "$PMG_MODEL_ROOT" \
  "$PMG_CKPT_ROOT" \
  "$PMG_DOWNLOAD_ROOT" \
  "$PMG_MANIFEST_ROOT"
```

先检查空间：

```bash
df -h /root/autodl-tmp
du -sh "$PMG_ROOT" 2>/dev/null || true
nvidia-smi
```

磁盘建议：

- baseline 最小闭环：至少预留 250–350 GB；
- 加 Isaac Sim 本地离线 assets：建议 500 GB–1 TB；
- TIC-VLA 全量 538 GB + 解压/缓存/训练输出：建议独立 1–1.5 TB；
- Hugging Face cache 和 `local_dir` 不能重复保存两份大数据；下载前检查 `HF_HOME`。

建议把缓存也放到数据盘：

```bash
export HF_HOME="$PMG_ROOT/cache/huggingface"
export PIP_CACHE_DIR="$PMG_ROOT/cache/pip"
export TORCH_HOME="$PMG_ROOT/cache/torch"
mkdir -p "$HF_HOME" "$PIP_CACHE_DIR" "$TORCH_HOME"
```

---

## 3. 中国境内服务器的统一回退规则

### 3.1 回退顺序

| 资源类型 | 第一选择 | 第二选择 | 最终安全回退 |
|---|---|---|---|
| GitHub repo | 官方 GitHub | 显式配置的可信 GitHub 代理 | 本地从官方源下载后 `rsync/scp` 上传 |
| Hugging Face 模型/数据 | `huggingface.co` | `hf-mirror.com` | ModelScope（仅存在同名官方模型时）或本地上传 |
| InternVL3-1B | Hugging Face 官方 | ModelScope `OpenGVLab/InternVL3-1B` | 本地上传 |
| Python 包 | PyPI 官方 | 清华 TUNA PyPI | 离线 wheelhouse 上传 |
| Conda 包 | defaults/conda-forge | 清华 TUNA Anaconda 镜像 | 导出环境包后上传 |
| Isaac Sim/Assets | NVIDIA 官方 CDN | `aria2c` 断点续传 | 本地从 NVIDIA 官方下载再上传 |

### 3.2 GitHub 镜像策略

不把未经验证的第三方 GitHub 代理写死到项目里。若 AutoDL 上 GitHub 不通，由用户提供当时可用且可信的代理前缀：

```bash
export GITHUB_PROXY_PREFIX='https://<trusted-proxy>/'
```

使用约定是“代理前缀 + 完整官方 URL”，例如：

```bash
git clone "${GITHUB_PROXY_PREFIX}https://github.com/ucla-mobility/TIC-VLA.git"
```

如果没有可信代理，最安全方法是在能访问 GitHub 的本机 clone 官方仓库，再上传到 AutoDL。不要使用来源不明的二次打包 zip。

### 3.3 Hugging Face 镜像策略

官方源：

```bash
export HF_ENDPOINT=https://huggingface.co
```

官方源连接失败时切换国内可访问镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

`hf-mirror.com` 是第三方镜像，不改变 repo id。下载后仍要校验文件清单、权重可加载性和关键文件 SHA-256。

如果 Xet/CAS 连接报错而普通文件可访问，可临时关闭 Xet 后续传：

```bash
export HF_HUB_DISABLE_XET=1
```

### 3.4 PyPI 与 Conda 镜像

单次 pip 使用清华源：

```bash
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple <package>
```

NVIDIA/Isaac 包不一定被普通镜像收录，必须保留 NVIDIA extra index：

```bash
python -m pip install \
  -i https://pypi.tuna.tsinghua.edu.cn/simple \
  --extra-index-url https://pypi.nvidia.com \
  <nvidia-or-isaac-package>
```

Conda 镜像只在官方 channel 超时后配置，参考清华 TUNA 当前的
[PyPI 帮助页](https://mirrors.tuna.tsinghua.edu.cn/help/pypi/)和
[Anaconda 帮助页](https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/)，不在本项目内永久覆盖用户全局 `.condarc`。

---

## 4. Stage 1A：TIC-VLA baseline 代码

官方仓库：<https://github.com/ucla-mobility/TIC-VLA>  
官方说明：仓库包含 model、supervised training、DynaNav assets 和 IsaacLab PPO scripts。

### 4.1 官方源获取

```bash
git clone --depth 1 \
  https://github.com/ucla-mobility/TIC-VLA.git \
  "$PMG_SRC_ROOT/TIC-VLA"
```

### 4.2 GitHub 不通时

```bash
git clone --depth 1 \
  "${GITHUB_PROXY_PREFIX}https://github.com/ucla-mobility/TIC-VLA.git" \
  "$PMG_SRC_ROOT/TIC-VLA"
```

仍失败时，在本机执行官方 clone，然后上传：

```bash
rsync -av --progress TIC-VLA/ \
  <autodl-user>@<autodl-host>:/root/autodl-tmp/ccfb-ticvla/src/TIC-VLA/
```

### 4.3 版本记录与验收

```bash
git -C "$PMG_SRC_ROOT/TIC-VLA" rev-parse HEAD \
  | tee "$PMG_MANIFEST_ROOT/TIC-VLA_COMMIT.txt"
git -C "$PMG_SRC_ROOT/TIC-VLA" status --short

test -d "$PMG_SRC_ROOT/TIC-VLA/ticvla"
test -d "$PMG_SRC_ROOT/TIC-VLA/DynaNav"
test -d "$PMG_SRC_ROOT/TIC-VLA/rl/fine_tuning"
test -f "$PMG_SRC_ROOT/TIC-VLA/tic-vla.yaml"
```

注意：不要在 upstream repo 内大面积直接改代码。PMG 实现在主项目的独立 `pmg_nav/` package 中，通过 wrapper/adapter 接入。

---

## 5. Stage 1A：InternVL3-1B 基础模型

Hugging Face 官方：<https://huggingface.co/OpenGVLab/InternVL3-1B>  
ModelScope 国内源：<https://modelscope.cn/models/OpenGVLab/InternVL3-1B>

### 5.1 Hugging Face 官方/镜像通用方法

先安装下载工具：

```bash
python -m pip install -U huggingface_hub
```

官方源或镜像只通过 `HF_ENDPOINT` 切换，命令不变：

```bash
export TICVLA_BASE_MODEL_PATH="$PMG_MODEL_ROOT/InternVL3-1B"

python - <<'PY'
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="OpenGVLab/InternVL3-1B",
    repo_type="model",
    local_dir=os.environ["TICVLA_BASE_MODEL_PATH"],
)
PY
```

若官方 Hugging Face 超时，执行：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

然后重新运行同一段 Python；`snapshot_download` 支持续传和缓存。

### 5.2 ModelScope 国内回退

```bash
python -m pip install -U modelscope

python - <<'PY'
import os
from modelscope import snapshot_download

snapshot_download(
    "OpenGVLab/InternVL3-1B",
    local_dir=os.environ["TICVLA_BASE_MODEL_PATH"],
)
PY
```

ModelScope 回退后必须确认目录格式与 TIC-VLA/Transformers 兼容；不能因为 repo 名相同就跳过加载测试。

### 5.3 验收

```bash
test -f "$TICVLA_BASE_MODEL_PATH/config.json"
find "$TICVLA_BASE_MODEL_PATH" -maxdepth 1 -type f -printf '%f\n' | sort \
  | tee "$PMG_MANIFEST_ROOT/InternVL3-1B_FILES.txt"
du -sh "$TICVLA_BASE_MODEL_PATH"
```

加载测试在 TIC-VLA conda 环境完成：

```bash
python - <<'PY'
import os
from transformers import AutoConfig

path = os.environ["TICVLA_BASE_MODEL_PATH"]
config = AutoConfig.from_pretrained(path, trust_remote_code=True)
print(type(config).__name__)
PY
```

---

## 6. Stage 1A：TIC-VLA checkpoint 与 DynaNav 子集

官方数据仓库：<https://huggingface.co/datasets/handsomeYun/TIC-VLA>  
许可证：CC-BY-4.0  
全量大小：约 **538 GB**  
组成：`SCAND/`、`GND/`、`DynaNav/`，以及仓库根目录的
`TIC-VLA-model.ckpt`。以下路径已按 TIC-VLA commit
`9fa6f8b66b9e121d5df5df071297bba8e5353ebb` 的官方 README 核对。

截至 2026-07-26，Stage 1 必需文件的官方压缩体积为：

- `TIC-VLA-model.ckpt`：1,938,501,547 bytes；
- `DynaNav/DynaNav_data.zip`：77,480,082,717 bytes；
- `DynaNav/DynaNav_json.zip`：25,904,780,595 bytes。

DynaNav 在 Hugging Face 上只发布为这两个 zip，不能用
`DynaNav/DynaNav_data/**` 或 `DynaNav/DynaNav_json/**` 选择内部文件。

### 6.1 下载前先查看文件与预计体积

```bash
python - <<'PY'
from huggingface_hub import HfApi

api = HfApi()
for path in api.list_repo_files("handsomeYun/TIC-VLA", repo_type="dataset"):
    if path == "TIC-VLA-model.ckpt" or path.startswith("DynaNav/"):
        print(path)
PY
```

如果当前 `hf` CLI 支持 dry-run，再执行：

```bash
hf download handsomeYun/TIC-VLA \
  --repo-type dataset \
  --include \
    'TIC-VLA-model.ckpt' \
    'DynaNav/DynaNav_data.zip' \
    'DynaNav/DynaNav_json.zip' \
  --dry-run
```

### 6.2 首轮最小下载：checkpoint + DynaNav

```bash
export TICVLA_DATA_ROOT="$PMG_DATA_ROOT/TIC-VLA"

python - <<'PY'
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="handsomeYun/TIC-VLA",
    repo_type="dataset",
    local_dir=os.environ["TICVLA_DATA_ROOT"],
    allow_patterns=[
        "TIC-VLA-model.ckpt",
        "DynaNav/DynaNav_data.zip",
        "DynaNav/DynaNav_json.zip",
    ],
)
PY
```

下载后解压到 Hugging Face README 规定的目录结构，并保留 zip 直至验收通过：

```bash
cd "$TICVLA_DATA_ROOT/DynaNav"
unzip -n DynaNav_data.zip
unzip -n DynaNav_json.zip
test -d DynaNav_data
test -d DynaNav_json
```

官方源失败时只需先设置：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

再运行同一段下载代码。不要因镜像失败而自动改成全量下载。

`DynaNav_data.zip` 单文件超过 50 GB。若镜像没有把 Xet 元数据传给
`huggingface_hub`，常规下载会报 `file is too large`，即使已经安装
`hf_xet` 也不会自动恢复。此时只对该文件使用镜像的 `resolve` 直链，
继续保留 repo id、commit 和文件字节数校验：

```bash
mkdir -p "$TICVLA_DATA_ROOT/DynaNav"
aria2c -c -x 8 -s 8 -k 1M --file-allocation=none \
  -d "$TICVLA_DATA_ROOT/DynaNav" \
  -o DynaNav_data.zip \
  'https://hf-mirror.com/datasets/handsomeYun/TIC-VLA/resolve/main/DynaNav/DynaNav_data.zip?download=true'

test "$(stat -c '%s' "$TICVLA_DATA_ROOT/DynaNav/DynaNav_data.zip")" \
  = 77480082717
```

该 URL 仍由镜像解析官方仓库对象，并重定向到 Hugging Face Xet 的内容
存储；`aria2c -c` 会保留并续传已有部分。不要删除 `.aria2` 控制文件或
已下载的数据片段。

### 6.3 将 checkpoint 放到固定目录

官方发布的 checkpoint 位于数据仓库根目录：

```bash
export TICVLA_CHECKPOINT_PATH="$TICVLA_DATA_ROOT/TIC-VLA-model.ckpt"
test -f "$TICVLA_CHECKPOINT_PATH"
```

无需复制一份 checkpoint；软链接到固定目录即可，避免浪费空间：

```bash
mkdir -p "$PMG_CKPT_ROOT"
ln -sfn "$TICVLA_CHECKPOINT_PATH" \
  "$PMG_CKPT_ROOT/ticvla.ckpt"
```

### 6.4 最小集合验收

```bash
test -d "$TICVLA_DATA_ROOT/DynaNav/DynaNav_json"
test -d "$TICVLA_DATA_ROOT/DynaNav/DynaNav_data"
test -f "$PMG_CKPT_ROOT/ticvla.ckpt"

find "$TICVLA_DATA_ROOT/DynaNav" -type f | wc -l
du -sh "$TICVLA_DATA_ROOT/DynaNav"
sha256sum "$TICVLA_CHECKPOINT_PATH" \
  | tee "$PMG_MANIFEST_ROOT/TIC-VLA_CHECKPOINT_SHA256.txt"
```

### 6.5 只有重做完整 SFT 时才下载全量

```bash
python - <<'PY'
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="handsomeYun/TIC-VLA",
    repo_type="dataset",
    local_dir=os.environ["TICVLA_DATA_ROOT"],
)
PY
```

全量下载前必须确认：

- 可用空间至少覆盖 538 GB 数据、缓存、训练输出和安全余量；
- `HF_HOME` 与 `local_dir` 不造成重复占用；
- 首轮 baseline 与 PMG 实验确实已经证明需要重新 SFT。

---

## 7. Stage 1B：Isaac Sim 5.0.0 与本地 Assets

官方安装页：<https://docs.isaacsim.omniverse.nvidia.com/5.0.0/installation/download.html>  
官方硬件要求：<https://docs.isaacsim.omniverse.nvidia.com/5.0.0/installation/requirements.html>  
TIC-VLA 固定使用 Isaac Sim 5.0.0。

重要兼容性：Isaac Sim 依赖 RTX ray tracing。NVIDIA 官方明确说明 A100/H100 这类无 RT Core 的 GPU 不受支持，因此本项目不能因算力强就租 H100/A100 跑 DynaNav 渲染。

### 7.1 Standalone Linux 包

官方直接地址：

```text
https://download.isaacsim.omniverse.nvidia.com/isaac-sim-standalone-5.0.0-linux-x86_64.zip
```

在服务器用断点续传：

```bash
cd "$PMG_DOWNLOAD_ROOT"
aria2c -c -x 8 -s 8 \
  'https://download.isaacsim.omniverse.nvidia.com/isaac-sim-standalone-5.0.0-linux-x86_64.zip'
```

如果 NVIDIA CDN 在 AutoDL 上不可达：在本机通过 NVIDIA 官方地址下载原文件，再上传；不使用第三方重打包 Isaac Sim。

解压和验证：

```bash
mkdir -p "$PMG_ROOT/isaacsim"
unzip -n \
  "$PMG_DOWNLOAD_ROOT/isaac-sim-standalone-5.0.0-linux-x86_64.zip" \
  -d "$PMG_ROOT/isaacsim"

cd "$PMG_ROOT/isaacsim"
./post_install.sh

export ISAAC_SIM_ROOT="$PMG_ROOT/isaacsim"
export ISAAC_SIM_PYTHON="$ISAAC_SIM_ROOT/python.sh"
"$ISAAC_SIM_PYTHON" -c \
  "from isaacsim import SimulationApp; print('Isaac Sim 5.0.0 ready')"
```

### 7.2 离线本地 Assets：网络不稳定时推荐

Isaac Sim 首次运行会在线读取 assets。中国服务器若访问慢，下载 NVIDIA 官方完整 assets 三个分片：

```bash
cd "$PMG_DOWNLOAD_ROOT"
aria2c -c -x 8 -s 8 \
  'https://download.isaacsim.omniverse.nvidia.com/isaac-sim-assets-complete-5.0.0.zip.001'
aria2c -c -x 8 -s 8 \
  'https://download.isaacsim.omniverse.nvidia.com/isaac-sim-assets-complete-5.0.0.zip.002'
aria2c -c -x 8 -s 8 \
  'https://download.isaacsim.omniverse.nvidia.com/isaac-sim-assets-complete-5.0.0.zip.003'
```

三个分片缺一不可。合并后解压：

```bash
cd "$PMG_DOWNLOAD_ROOT"
cat \
  isaac-sim-assets-complete-5.0.0.zip.001 \
  isaac-sim-assets-complete-5.0.0.zip.002 \
  isaac-sim-assets-complete-5.0.0.zip.003 \
  > isaac-sim-assets-complete-5.0.0.zip

mkdir -p "$PMG_DATA_ROOT/isaacsim_assets"
unzip -n isaac-sim-assets-complete-5.0.0.zip \
  -d "$PMG_DATA_ROOT/isaacsim_assets"
```

最终 asset root 应同时包含 `NVIDIA` 和 `Isaac`：

```bash
test -d "$PMG_DATA_ROOT/isaacsim_assets/Assets/Isaac/5.0/NVIDIA"
test -d "$PMG_DATA_ROOT/isaacsim_assets/Assets/Isaac/5.0/Isaac"
```

根据 NVIDIA 官方说明，在 Isaac Sim kit 配置中设置：

```toml
[settings]
persistent.isaac.asset_root.default = "/root/autodl-tmp/ccfb-ticvla/data/isaacsim_assets/Assets/Isaac/5.0"
```

注意：TIC-VLA 自带的 DynaNav local scene assets 与 NVIDIA 通用 assets 不是同一项，两者都保留原目录结构。

---

## 8. Stage 2：Physical Memory Graph 感知资源

Stage 2 先执行 annotation gate：如果 DynaNav/Isaac Sim 能直接提供可靠 object identity、pose、depth 和 segmentation，就优先使用 simulator oracle 建立 PMG，不必立即加入重型 detector。

### 8.1 GroundingDINO：条件必需

官方仓库：<https://github.com/IDEA-Research/GroundingDINO>  
官方 Swin-T checkpoint：<https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth>

获取代码：

```bash
git clone --depth 1 \
  https://github.com/IDEA-Research/GroundingDINO.git \
  "$PMG_SRC_ROOT/GroundingDINO"
```

GitHub 不通时同样使用 `${GITHUB_PROXY_PREFIX}`，或者本机从官方源下载后上传。

权重：

```bash
mkdir -p "$PMG_MODEL_ROOT/GroundingDINO"
aria2c -c -x 8 -s 8 \
  -d "$PMG_MODEL_ROOT/GroundingDINO" \
  'https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth'
```

验收：

```bash
git -C "$PMG_SRC_ROOT/GroundingDINO" rev-parse HEAD \
  | tee "$PMG_MANIFEST_ROOT/GroundingDINO_COMMIT.txt"
sha256sum "$PMG_MODEL_ROOT/GroundingDINO/groundingdino_swint_ogc.pth" \
  | tee "$PMG_MANIFEST_ROOT/GroundingDINO_SHA256.txt"
```

用途边界：GroundingDINO 只负责开放词汇 object boxes/labels；metric relation、3D pose、reachability 和 uncertainty 仍由 PMG 的 depth projection、geometry 和 belief update 负责。

### 8.2 SAM 2.1：可选，不是 MVP 硬依赖

官方仓库：<https://github.com/facebookresearch/sam2>  
建议先用 `sam2.1_hiera_tiny` 或 `small`，而不是 large，以减少显存和延迟。

```bash
git clone --depth 1 \
  https://github.com/facebookresearch/sam2.git \
  "$PMG_SRC_ROOT/sam2"
```

checkpoint 获取以 repo 内官方脚本和 README 当时列出的 Meta 官方 URL 为准：

```bash
cd "$PMG_SRC_ROOT/sam2/checkpoints"
./download_ckpts.sh
```

如果 GitHub/Meta 下载不通，先查看脚本中的官方 checkpoint URL，再通过可信代理或本地官方源下载上传。不要从网盘获取无法校验来源的模型。

加入 SAM 2 的触发条件：box 投影的 object association 明显不稳定，或 PMG ablation 需要 mask-level geometry。否则 Stage 2 MVP 保持 GroundingDINO box + depth。

---

## 9. Stage 3：IsaacLab v2.2.1 与 PPO

官方仓库：<https://github.com/isaac-sim/IsaacLab/tree/v2.2.1>  
TIC-VLA upstream 明确要求 `IsaacLab v2.2.1 + Isaac Sim 5.0.0`。

### 9.1 获取固定 tag

```bash
git clone --branch v2.2.1 --depth 1 \
  https://github.com/isaac-sim/IsaacLab.git \
  "$PMG_SRC_ROOT/IsaacLab"
```

GitHub 不通时：

```bash
git clone --branch v2.2.1 --depth 1 \
  "${GITHUB_PROXY_PREFIX}https://github.com/isaac-sim/IsaacLab.git" \
  "$PMG_SRC_ROOT/IsaacLab"
```

### 9.2 版本和兼容性验收

```bash
git -C "$PMG_SRC_ROOT/IsaacLab" describe --tags --always
git -C "$PMG_SRC_ROOT/IsaacLab" rev-parse HEAD \
  | tee "$PMG_MANIFEST_ROOT/IsaacLab_v2.2.1_COMMIT.txt"

export ISAACLAB_ROOT="$PMG_SRC_ROOT/IsaacLab"
ln -sfn "$ISAAC_SIM_ROOT" "$ISAACLAB_ROOT/_isaac_sim"
```

安装步骤严格跟 TIC-VLA `rl/README.md`，不要自行升级到最新 IsaacLab：

```bash
cd "$ISAACLAB_ROOT"
./isaaclab.sh --conda tic-vla
conda deactivate
conda activate tic-vla

python -m pip install --no-build-isolation flatdict==4.0.1
python -m pip install -e source/isaaclab --no-deps
python -m pip install -e source/isaaclab_assets --no-deps
python -m pip install -e source/isaaclab_tasks --no-deps
python -m pip install -e source/isaaclab_rl --no-deps
python -m pip install -e source/isaaclab_mimic --no-deps
```

Stage 3 不需要新增外部训练数据。PMG 的 tool-policy demonstrations、oracle labels 和 PPO rollouts 均在 DynaNav/Isaac Sim 中在线生成并保存到 `$PMG_ROOT/outputs`。

---

## 10. 条件 fallback：Habitat-Lab + HM3D/HM3DSem

只有当 Stage 2 annotation gate 证明 DynaNav 无法提供足够的 object identity/pose、metric-relation oracle 或可控 episode 时才启用。它不会替换 TIC-VLA baseline，只增加 TIC-VLA observation/action adapter。

需要的资源：

- Habitat-Lab：<https://github.com/facebookresearch/habitat-lab>
- Habitat-Sim：<https://github.com/facebookresearch/habitat-sim>
- HM3D/HM3DSem：按 Habitat/Matterport 官方许可页面申请和下载；数据受许可约束，不能用非官方镜像重新分发。

获取原则：

1. 先记录 gate 失败证据；
2. 再按照 Habitat 官方文档选定彼此兼容的 Habitat-Lab/Habitat-Sim release；
3. HM3D 必须使用获授权的官方下载链接；
4. 若官方数据 URL 在境内不可达，在有权限的本机下载后加密上传，不能从公开网盘找替代包；
5. 记录数据 license、split 版本和 scene manifest。

此 fallback 当前状态：**不下载、不占磁盘、不进入 Stage 1。**

---

## 11. Python 环境、系统包和辅助材料

### 11.1 系统工具

```bash
sudo apt-get update
sudo apt-get install -y \
  aria2 \
  build-essential \
  cmake \
  git \
  git-lfs \
  rsync \
  unzip
git lfs install
```

如果 AutoDL 镜像没有 sudo 权限，优先选择已包含编译工具的基础镜像，或联系 AutoDL 管理权限；不要把系统二进制从不明网盘复制进环境。

### 11.2 TIC-VLA 环境声明

环境配置以 upstream repo 中以下文件为准：

- `tic-vla.yaml`
- `pyproject.toml`
- `requirements-train.txt`
- `requirements-test.txt`

获取代码后执行：

```bash
cd "$PMG_SRC_ROOT/TIC-VLA"
conda env create -n tic-vla -f tic-vla.yaml
conda activate tic-vla
python -m pip install -e .
```

不要在获取资源手册中复制一份固定 requirements 并长期维护；upstream commit、环境文件和 `pip freeze` 一起构成可复现版本。

### 11.3 论文与文档材料

必须保存链接，不需要下载大型额外附件：

- TIC-VLA paper：<https://arxiv.org/abs/2602.02459>
- TIC-VLA code：<https://github.com/ucla-mobility/TIC-VLA>
- TIC-VLA dataset/checkpoint：<https://huggingface.co/datasets/handsomeYun/TIC-VLA>
- InternVL3-1B：<https://huggingface.co/OpenGVLab/InternVL3-1B>
- Isaac Sim 5.0 docs：<https://docs.isaacsim.omniverse.nvidia.com/5.0.0/>
- IsaacLab v2.2.1 docs：<https://isaac-sim.github.io/IsaacLab/v2.2.1/>
- GroundingDINO：<https://github.com/IDEA-Research/GroundingDINO>
- SAM 2：<https://github.com/facebookresearch/sam2>

### 11.4 非必需 API

TIC-VLA upstream 的 raw-data annotation 脚本可调用 OpenAI API 生成 instructions/CoT，但使用 released dataset 跑 baseline 不需要该 API。本项目 Stage 2 首版约束指令优先用 simulator template/oracle 生成，因此不把外部 API 作为 baseline 阻塞条件，也不在项目文件中保存任何 API key。

---

## 12. 完整性、版本和安全校验

每次获取资源后都要创建 manifest：

```bash
date -Iseconds | tee "$PMG_MANIFEST_ROOT/FETCH_TIME.txt"
nvidia-smi | tee "$PMG_MANIFEST_ROOT/NVIDIA_SMI.txt"
conda env export -n tic-vla --no-builds \
  > "$PMG_MANIFEST_ROOT/tic-vla-environment.yml"
conda run -n tic-vla python -m pip freeze \
  > "$PMG_MANIFEST_ROOT/tic-vla-pip-freeze.txt"
```

必须记录：

- TIC-VLA commit；
- IsaacLab tag 和 commit；
- GroundingDINO/SAM2 commit（若启用）；
- Hugging Face/ModelScope repo id；
- checkpoint SHA-256；
- 关键目录文件数和 `du -sh`；
- 下载源是 official、HF mirror、ModelScope 还是本机上传；
- 数据许可证和访问日期。

镜像安全规则：

- 代码镜像 clone 后仍检查 git commit 和关键文件；
- checkpoint 至少记录 SHA-256，并实际执行一次 loader test；
- 不执行第三方页面提供的 `curl ... | bash`；
- 不把 token、密码、API key 写入 `.env` 后提交；
- 对有许可的数据（例如 HM3D）不进行二次公开分发；
- 如官方与镜像内容不一致，以官方仓库 metadata 和可加载性为准，停止训练并重新取资源。

---

## 13. 后续在 AutoDL 上的实际获取顺序

严格按以下 checklist 执行：

- [x] 1. 创建统一目录，确认数据盘和缓存位置；
- [x] 2. 测试 GitHub、Hugging Face、NVIDIA CDN、PyPI；
- [x] 3. clone TIC-VLA，记录 commit；
- [x] 4. 创建 `tic-vla` 环境；
- [x] 5. 下载 InternVL3-1B；
- [x] 6. 仅下载 TIC-VLA checkpoint + DynaNav 子集；
- [x] 7. 跑 open-loop 单样本及固定 subset 推理；
- [x] 8. 验证 Isaac Sim 5.0 不兼容后安装 Isaac Sim 6.0.1 compatibility runtime；
- [x] 9. 在线 hospital scene 可正常加载，本轮无需下载 local assets packs；
- [x] 10. 跑通并独立复验 DynaNav closed-loop 单 episode；
- [ ] 11. 进入 Stage 2 annotation gate；
- [ ] 12. gate 需要时再获取 GroundingDINO，SAM2 仍保持可选；
- [ ] 13. Stage 2 无 RL PMG 验收后再 clone IsaacLab v2.2.1；
- [ ] 14. 进入 PPO/PMG tool policy；
- [ ] 15. 只有论文实验确定需要重做 SFT 时，才下载 538 GB 全量 TIC-VLA 数据；
- [ ] 16. 只有 DynaNav gate 失败时，才申请 Habitat/HM3D fallback。

任何一步若官方源失败：先记录错误日志，再切换本节规定的镜像路径；不得无记录地混用多个来源。
