# Simple RVC Studio 工作交接记录

更新时间：2026-04-17  
当前提交：`386ef68 Improve Simple RVC long audio workflow`

这份文档用于在另一台电脑继续接手当前项目，重点记录本轮对 `tools/simple_rvc_flask.py` 与长音频处理链路的改动、验证结果和仍需注意的问题。

## 环境与启动

当前项目目录：

```bash
/Users/yanwei/src/Retrieval-based-Voice-Conversion-WebUI
```

项目使用已有的 `uv` 虚拟环境：

```bash
source .venv/bin/activate
python tools/simple_rvc_flask.py
```

服务地址：

```text
http://127.0.0.1:7867/
```

系统依赖中已用到：

```bash
ffmpeg
node / npm / npx
```

其中 `node` 是为了用 Playwright 检查页面布局时通过 Homebrew 安装的。运行 RVC 本身主要依赖 Python 环境与 ffmpeg。

## Git 状态

远端已切为 SSH：

```text
origin git@github.com:yanwei/Retrieval-based-Voice-Conversion-WebUI.git
upstream git@github.com:RVC-Project/Retrieval-based-Voice-Conversion-WebUI.git
```

上一轮功能改动已提交并推送：

```text
386ef68 Improve Simple RVC long audio workflow
```

本轮还新增了本交接文档，建议另一台电脑直接：

```bash
git pull origin main
```

## 已忽略的本地/大文件

`.gitignore` 已覆盖主要无用或本机文件：

```text
/.playwright-cli/
/outputs
/downloads
.venv
/TEMP
/assets/weights/*
/assets/indices/*
/assets/uvr5_weights/*
/assets/rmvpe/*
/logs
```

注意：模型权重、索引、UVR 权重仍需在目标机器本地存在，不会随 Git 同步。

关键本地文件包括：

```text
assets/weights/Myaz.pth
assets/weights/haoshengyin.pth
assets/weights/mygf-jack.pth
assets/indices/haoshengyin.index
assets/indices/mygf-jack.index
assets/uvr5_weights/HP5_only_main_vocal.pth
assets/uvr5_weights/HP3_all_vocals.pth
assets/uvr5_weights/HP2_all_vocals.pth
assets/rmvpe/rmvpe.pt
```

`Myaz.pth` 当前似乎没有对应 index，因此代码已处理空 index，不会再把空 index 误解析为 `assets/indices` 目录。

## 页面改动

主要文件：

```text
tools/templates/simple_rvc_flask.html
tools/static/simple_rvc_flask.js
tools/simple_rvc_flask.py
```

页面名称仍是 `Simple RVC Studio`，长音频页做了这些调整：

- 页面宽度改成充分利用屏幕，不再固定窄宽度。
- “原始音频文件”和原始音频播放器放在同一行。
- 长音频参数区更紧凑，但后来根据实际 Playwright 检查放松了垂直间距。
- 男女输入分路左右并排，每个分路内部三列两行。
- 右侧状态区保留日志、转换后人声、最终成品播放器。
- 默认男声模型改为 `Myaz.pth`。
- 默认女声模型改为 `haoshengyin.pth`。
- 长音频页新增“人声分离模型”选择：`auto`、`HP5_only_main_vocal`、`HP3_all_vocals`、`HP2_all_vocals`。

模型默认参数：

```text
mygf-jack.pth:
  f0_up_key: 0
  index_rate: 0.00
  rms_mix_rate: 0.86

haoshengyin.pth:
  f0_up_key: 0
  index_rate: 0.75
  protect: 0.25
  rms_mix_rate: 1.25

Myaz.pth:
  f0_up_key: 0
  index_rate: 0.75
  protect: 0.25
  rms_mix_rate: 1.25
```

## MPS / TorchScript 修复

文件：

```text
infer/lib/infer_pack/commons.py
```

修复过 `commons.fused_add_tanh_sigmoid_multiply(...)` 在 Apple MPS 上报错：

```text
NotImplementedError: Unknown device for graph fuser
```

处理方式是避免走会触发 graph fuser 的 TorchScript 路径，改成普通 PyTorch op 路径。

## 长音频处理链路改动

核心文件：

```text
tools/process_mixed_long_audio.py
```

### UVR 分离

之前遇到的问题：

- MPS 上 UVR 可能输出饱和/刺耳噪声。
- 多任务或重跑时 `/tmp/*.reformatted.wav` 可能冲突。
- 长音频直接跑 UVR 不稳定。
- `HP3_all_vocals` 和 `HP5_only_main_vocal` 对不同音频效果不同。

目前处理：

- UVR 分离强制走 CPU。
- 每个 job 使用独立 `uvr_tmp`，避免临时文件冲突。
- 超过 90 秒的音频按 30 秒 chunk 分离后拼回。
- 新增 `UVR_MODEL = "auto"`，自动在 HP5 / HP3 间选择。
- Auto 先用 60 秒 sample 跑 HP5 与 HP3 做评分，然后对完整音频重新跑选中的模型。

重要修复：

之前 Auto 在短音频上误把 60 秒 sample 的分离结果当正式结果，导致 `1:04` 音频被截成约 `59s`。现在 sample 只用于评分，正式输出一定基于完整原音频。

### Auto UVR 选择规则

当前候选：

```python
UVR_AUTO_CANDIDATES = ("HP5_only_main_vocal", "HP3_all_vocals")
```

判断逻辑大意：

- 如果 HP3 vocal 覆盖率非常高，像是把音乐也当成人声；
- 且 HP5 仍保留足够人声；
- 且 HP5 明显降低了疑似音乐混入；
- 则选 HP5。
- 否则选 HP3。

这次针对 `page_313682_1438343.mp3` 修过一个临界误判：HP3 覆盖率约 `0.913`，HP5 约 `0.763`，旧规则因差值卡在 `0.1499` 而误选 HP3，导致 5 秒开始音乐被当人声转换。现在该情况会选 HP5。

如果页面上仍出现音乐被当人声，临时手动选择：

```text
HP5_only_main_vocal
```

通常会比 auto 更保险。

## 分段与直通处理

之前遇到的问题：

- 很多 `0.x` 秒短直通片段没有被并入前后。
- 短直通插在转换结果中，容易形成原声碎片或边界刺啦声。

当前处理：

- `low_voice` 的短直通片段在 `1.5s` 以内会尽量吸收到前后已判定人声上下文。
- 只在与前后间隔不超过 `3s` 时吸收，避免把远处噪声硬合并。
- 吸收后再合并上下文片段。
- 片段输出前加 `25ms` 边缘 fade，减少拼接边界噪声。

相关常量：

```python
SHORT_PASSTHROUGH_CONTEXT_SEC = 1.50
SHORT_PASSTHROUGH_CONTEXT_GAP_SEC = 3.00
SEGMENT_EDGE_FADE_SEC = 0.025
```

## 男女声判定

之前问题：

- `page_313682_1438343.mp3` 中 36 秒后的段落听感是男声，但旧阈值把 median F0 约 `188.5Hz` 的段落判成女声。

当前处理：

```python
DECISION_F0_HZ = 190.0
```

因此 36 秒后的 `188.5Hz` 段会判为男声，走 `Myaz.pth`。

注意：这仍是启发式阈值，不是完美说话人识别。如果后续遇到更复杂的男女混合，需要考虑：

- 开启高精度 speaker embedding；
- 或增加手动分段/手动 route 覆盖；
- 或基于音频文件维护 per-file overrides。

## 已测试音频与结果

### `page_313682_1438343.mp3`

原始路径：

```text
/Users/yanwei/Downloads/RVC Sample/page_313682_1438343/page_313682_1438343.mp3
```

重要测试目录：

```text
outputs/simple_rvc_flask/jobs/codex_page_313682_length_gender_fix
```

验证结果：

```text
原音频:      64.419s
vocals.wav: 64.414s
final_mix:  64.414s
```

长度问题已修复，不再被截成 60 秒。

该次分段：

```text
0.791-2.147s    female
14.816-24.206s  female
24.429-31.357s  female
36.422-53.428s  male
```

后续又发现页面任务中 Auto 误选 HP3，导致 5 秒开始音乐被当人声。已调整 Auto 规则，让同类指标选择 HP5。

### `unit_083705.mp3`

原始路径：

```text
/Users/yanwei/Downloads/RVC Sample/unit_083705.mp3
```

之前测试 HP3 效果明显优于 HP5。Auto smoke test 曾选择 HP3，因此不要简单全局固定 HP5。

## 已知风险与下一步建议

1. `auto` 仍是启发式选择，不可能 100% 判断 HP5/HP3。遇到音乐被当人声时，优先手动选 `HP5_only_main_vocal` 复测。

2. `Myaz.pth` 当前没有 index，代码已避免空 index 误读目录，但没有 index 可能影响音色稳定性。若模型提供者有 `Myaz.index`，建议补到 `assets/indices/`。

3. 页面默认 `rms_mix_rate=1.25` 是按模型推荐参数设置，但部分音频可能放大瑕疵。如果转换后有爆音或刺耳声，可临时降到 `0.8-1.0` 试。

4. 高精度 speaker embedding 会尝试加载 SpeechBrain 模型，可能联网或很慢。页面默认关闭是合理的。需要更准的男女/说话人区分时再打开。

5. 生成文件都在 `outputs/simple_rvc_flask/jobs/`，已被 Git ignore。换机器不会同步这些历史测试结果。

## 常用排查命令

查看最近任务分段：

```bash
find outputs/simple_rvc_flask/jobs -maxdepth 2 -name segments.csv -print0 | xargs -0 ls -t | head
column -s, -t < outputs/simple_rvc_flask/jobs/<job>/segments.csv | sed -n '1,80p'
```

查看 Auto UVR 选择：

```bash
cat outputs/simple_rvc_flask/jobs/<job>/uvr_auto_selection.json
```

检查音频时长：

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
import soundfile as sf
for p in [
    Path('outputs/simple_rvc_flask/jobs/<job>/vocals.wav'),
    Path('outputs/simple_rvc_flask/jobs/<job>/converted_vocals.wav'),
    Path('outputs/simple_rvc_flask/jobs/<job>/final_mix.wav'),
]:
    info = sf.info(p)
    print(p.name, round(info.duration, 3), info.samplerate, info.channels)
PY
```

检查某个时间段是否被转进 converted vocals：

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
import soundfile as sf
import numpy as np

audio, sr = sf.read('outputs/simple_rvc_flask/jobs/<job>/converted_vocals.wav', always_2d=True, dtype='float32')
for start, end in [(0, 3), (3, 8), (5, 10), (10, 14), (14, 16)]:
    chunk = audio[int(start * sr):int(end * sr)]
    rms = float(np.sqrt(np.mean(chunk * chunk))) if chunk.size else 0.0
    peak = float(np.max(np.abs(chunk))) if chunk.size else 0.0
    print(f'{start}-{end}s rms={rms:.5f} peak={peak:.5f}')
PY
```

## 当前建议工作流

1. 拉取最新代码。
2. 确认本机模型与 UVR 权重存在。
3. 启动 `python tools/simple_rvc_flask.py`。
4. 对新音频先用 `auto` 跑。
5. 如果音乐被当人声，改选 `HP5_only_main_vocal` 重跑。
6. 如果人声缺失或被分离太少，改选 `HP3_all_vocals` 重跑。
7. 查看 `segments.csv` 判断男女 route 是否合理。
8. 如某段性别不对，优先调整阈值或后续增加手动 route override。
