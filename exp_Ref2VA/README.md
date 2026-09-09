# Ref2VA 批量推理

80张 `data_h3/cat_ids/` 首帧 × `drag_ear` 动作 × seed `0`，当前metadata共80条任务，保留原任务编号和文件名。复用现有8卡调度器，两个四卡副本（TP2 × Ulysses2）共享队列，结果写入 `exp_Ref2VA/videos/drag_ear/`。

## 运行

从 `zonghui-liu-opt/MiniMax-H3` 的 `data_pipeline` 分支拉取代码，在工程根目录执行：

```bash
# 全量输入检查：不创建缓存或请求预览，不访问模型服务
bash infer_ref2va_8gpu.sh --dry-run

# 自动准备静音缓存、启动服务、推理下载、关闭本次服务
bash infer_ref2va_8gpu.sh --start-servers \
  --model-path /srv/workspace/Kirin_AI_DataLake/models/MiniMax-H3
```

无需携带静音副本、工程包或源码快照。Git保存原始 `Ref_drag_ear.mp4`，首次实际推理自动在 `prepared/` 创建无音轨缓存，后续校验哈希并复用；默认metadata已提交，可直接运行。GPU若仍被原FL2VA服务占用，先在其服务终端停止。脚本只管理自己启动的进程。

常驻服务：终端A执行 `bash serve_ref2va_8gpu.sh --model-path /path/to/MiniMax-H3`，终端B执行 `bash infer_ref2va_8gpu.sh`。`--limit 3` 可先运行当前CSV前三条任务，随后整批运行会跳过已完成结果。

| 副本 | GPU | HTTP | ZMQ | master | scheduler |
| --- | --- | --- | --- | --- | --- |
| 1 | 0,1,2,3 | 30110 | 30111 | 31110 | 32110 |
| 2 | 4,5,6,7 | 30112 | 30113 | 31111 | 32111 |

默认每服务一个在途任务。可用 `--server-urls` 连接已有Ref2VA服务，或用 `--gpu-groups`、`--tp-size`、`--ulysses-degree` 和端口参数调整部署。

## 指定原生生成分辨率

推理时传 `--resolution WIDTHxHEIGHT`，例如 `480x832`。参数控制模型实际生成画布，不做输出缩放；不传则保留原来的768短边规则。宽高须为32的倍数，并能由H3支持的比例和画布规则精确推导，无法精确表示的尺寸会直接报错。

第一次切换低分辨率时，在原服务终端按Ctrl-C停止服务，然后执行：

```bash
# 终端A：显式开启自定义分辨率，再启动两个四卡副本
bash serve_ref2va_8gpu.sh --resolution 480x832

# 终端B：指定实际推理尺寸；新输出目录保留此前768结果
bash infer_ref2va_8gpu.sh --resolution 480x832 --output-dir exp_Ref2VA/480x832 --retry-failed
```

结果位于 `exp_Ref2VA/480x832/videos/drag_ear/`，文件名继续沿用metadata。也可一体化运行 `bash infer_ref2va_8gpu.sh --start-servers --resolution 480x832 --output-dir exp_Ref2VA/480x832`。服务启动参数用于放开自定义尺寸，实际每条请求的尺寸由推理命令指定；之后改变支持的分辨率无需再次重启服务。

较早的H3接口在两处硬编码要求短边768。只有显式指定 `--resolution` 并启动服务时，入口才会在导入这两个模块时，将已知限制在内存中改为正整数检查。该处理在服务主进程和所有GPU工作进程中生效，不写入SGLang安装文件或字节码缓存；只读环境也可使用，无需sudo、修改目录权限或安装依赖。保留默认值、比例限制、32像素对齐和像素上限，不检查版本、不生成补丁包。`--dry-run` 仅预览命令。此前失败的任务可用 `--retry-failed` 继续。

H3忽略 `target.width/height`，且 `short_edge=480, aspect_ratio=9:16` 会对齐为480×864。入口为480×832发送 `short_edge=468, aspect_ratio=9:16`，让服务原生得到精确画布。分辨率计入请求指纹，拒绝误复用其他尺寸的任务；下载后也核实实际MP4宽高，不匹配不会标记完成。接口规则见[官方shape resolver](https://github.com/sgl-project/sglang/blob/v0.5.19/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/resolved_plan.py)。低分辨率减少目标视频token，但参考视频编码和参考token仍有计算成本，实际加速与效果需在GPU上验证。

## 沿用现有环境

在已经跑通FL2VA的环境执行，启动方式复用 `serve_smoke_8gpu.sh` 的调度器，模型参数使用 `--model-variant ref2va`。不做包版本、权重头或编码器预检，不要求自定义服务名称；SGLang直接加载模型并报告实际错误。无需为本次修正安装指定版本或搬运wheel包。

`--model-path` 传含完整 `Ref2VA/` 分区的模型根目录，不能传 `Ref2VA` 子目录或FL2VA分区。入口保持Hub离线模式。批量客户端使用Python标准库和现有 `ffmpeg`、`ffprobe`；静音缓存通过 `-c:v copy -an` 复制视频轨道，不需要libx264重新编码。只启动server不处理这些素材。

## 输入、提示词与命名

永久维护文件只有本README、`cat_catalog.csv`、`motions.json`、`metadata.csv`、原动作视频及 `prompts/` 下的版本化提示词。CSV通过相对 `prompt_file` 引用单份提示词，保留文本/源视频哈希，不重复存储提示词正文。

文件名为 `{id}_{cat_id}_{motion_id}_seed-{seed}.mp4`，例如：

```text
videos/drag_ear/000_00-orange-shorthair-mackerel-tabby_02-pull-left-ear_seed-0.mp4
```

`drag_ear.v1` 使用[官方六段格式](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md)。参考动作是猫用自己的左前爪（画面右侧）触左耳/头侧、向下擦脸、落爪恢复坐姿。身份、自然耳型、毛色/皮肤和场景来自首帧；无尾帧锁定。

请求固定 `task=ref2va`，默认按顺序提交猫图 `image/reference` 和静音 `video/reference`，分别对应 `<Picture 1>` 的身份/场景和 `<Video 1>` 的动作。提示词引导开头构图，但默认不硬锁第0帧。原提示词、80只猫、当前seed=0的metadata及输出命名继续使用。

需要接口层首帧约束且服务支持混合条件时，推理命令可显式加 `--lock-first-frame`：额外提交同图 `image/keyframe, frame_index=0`，保留身份reference以提供 `<Picture 1>`。不检测版本、不修改SGLang、不自动降级；服务不支持时保留实际错误。切换此选项会改变请求指纹，避免误复用旧结果。

视频含原音轨会自动引入音频参考，因此去除音轨。源猫图480×832采用最近合法比例9:16；15:26不被接受，auto会回落为横屏。接口来源：[SGLang H3文档](https://docs.sglang.io/cookbook/diffusion/MiniMax/MiniMax-H3)。

默认50步、768短边（可用 `--resolution` 指定）、4秒请求、24fps、flow_shift=12、audio_flow_shift=3。模型帧数按17n+5对齐，实际MP4可能长于4秒；保留原始输出。用户已反馈原768分辨率生成效果良好；480×832尚待实际GPU验证，metadata中的历史提示词状态保持原样。

新增或修改动作：保存 `Ref_*.mp4`，查看实际动作后编写独立版本提示词，在 `motions.json` 登记，然后执行：

```bash
python3 scripts/prepare_ref2va_metadata.py
# 可选筛选，保留相同任务网格下的编号
python3 scripts/prepare_ref2va_metadata.py --cat-ids 00 02 38 \
  --motion-ids drag_ear --seeds 0 10000 20000 \
  --output exp_Ref2VA/metadata_subset.csv
bash infer_ref2va_8gpu.sh --metadata exp_Ref2VA/metadata_subset.csv
```

准备metadata默认不生成媒体副本，需要提前生成缓存可加 `--prepare-media`。未知新动作不会自动套用已有提示词。旧版内联prompt CSV仍可读取。

## 运行产物与恢复

`prepared/`、`videos/`、`states/`、`logs/`、`services/`、`history/` 及结果清单均由 `.gitignore` 排除。默认不写逐条请求预览，调试时显式加 `--write-requests`。不生成工程压缩包、源码副本或单独验证报告。

保留必要的逐任务state用于断点恢复：同请求跳过已下载视频；未完成ID回原服务恢复；服务重启导致ID返回404才重提。提示词、源素材或采样参数变化会拒绝复用旧结果。更换模型权重使用新输出目录，因为不会哈希整套权重。

`--retry-failed` 重提明确失败任务；轮询超时保留ID。POST响应丢失标记 `submission_unknown`，检查服务日志后再决定重跑，避免重复占用GPU。`--force` 会把旧结果归档至忽略的 `history/` 再重新生成。输出目录使用进程锁。成功退出0、部分失败1、配置错误2、中断130。

回归测试：`python3 -m unittest discover -s tests -q`。模拟服务测试覆盖默认双参考请求、可选首帧约束、服务启动、命名、双副本、恢复与失败重试，不代表真实H100生成质量。
