# Ref2VA 批量推理

复用已在内网 H100×8 跑通的 `infer_ref2va_8gpu.sh`，不新增推理入口或环境依赖。当前共 **12 个参考视频 × 80 个猫咪 ID × seed 0 = 960 条任务**。包含原 `drag_ear` 对照和 11 个新增参考；镜像视频独立计数。新增提示词已按实际视频审阅并使用官方 `h3-prompt-writing` 六段格式，生成效果均待内网模型验证。

## 按首帧猫咪 ID 定制 prompt（cat_identity.v1，待模型验证）

逐张审阅80张首帧，分别描述毛色、花纹分布、眼色、脸型、耳形、被毛/裸肤和可见尾部特征。身份来自实际图像，不仅依据品种名。例如00保留橘色虎斑纹，38保留灰色裸肤和大耳；07与31的异瞳左右不同，镜像动作也不翻转这些身份特征。

只新增一份 `prompts/cat_identity.v1.json`（80份身份描述及共用组合模板），复用12份现有动作prompt。生成器与推理入口使用同一个组合函数，将具体身份写入主体定义、保留策略、开场、动作中段和结尾，保留动作时间线、玩具、方向和结束姿态。实际提交的是每只猫各自的完整英文六段prompt；没有复制960份文本文件，也没有新增推理入口或环境依赖。

| 文件 | 用途 |
| --- | --- |
| `prompts/cat_identity.v1.json` | 80只猫的独立描述、身份锚点、所审阅猫图SHA256 |
| `metadata_identity_v1.csv` | 80猫 × 12个已登记动作 × seed 0，共960条 |
| `metadata_identity_v1_smoke.csv` | 猫咪00覆盖12动作，任务名与新全量表一致 |

在内网工程根目录执行，以下复用已运行的Ref2VA服务；需要启动服务则追加 `--start-servers --model-path /path/to/MiniMax-H3`。先确认下节记录的服务端keyframe兼容问题已经解决，再提交锁帧实验。

```bash
# 检查指定猫的12个动作输入；00可换成38或其他ID，也可传多个ID
bash infer_ref2va_8gpu.sh --metadata exp_Ref2VA/metadata_identity_v1.csv \
  --cat-ids 00 --lock-first-frame \
  --output-dir outputs/ref2va_identity_v1_lock --dry-run

# 实际生成指定猫；保留原对照使用的resolution和其他采样参数
bash infer_ref2va_8gpu.sh --metadata exp_Ref2VA/metadata_identity_v1.csv \
  --cat-ids 00 --lock-first-frame \
  --output-dir outputs/ref2va_identity_v1_lock

# 效果确认后跑全量；保持参数与目录一致，跳过已完成的指定猫任务
bash infer_ref2va_8gpu.sh --metadata exp_Ref2VA/metadata_identity_v1.csv \
  --lock-first-frame --output-dir outputs/ref2va_identity_v1_lock
```

直接使用 `metadata_identity_v1_smoke.csv` 也可预览00号猫。`--cat-ids 00 38` 同时测两只；追加 `--motion-ids cat_teaser2 drag_yarn_ball_and_catch` 只测指定动作。推理筛选保持CSV中的任务名和seed，`--limit N` 在筛选后取前N条。要检查完整组合prompt，可在dry-run命令中显式追加 `--write-requests --limit 1`，阅读 `outputs/ref2va_identity_v1_lock/requests/<motion_slug>/*.json` 的 `prompt`；默认dry-run不写文件。

新metadata的 `prompt_file` 引用原动作正文，`identity_prompt_file` 引用身份配置，按 `cat_id` 匹配；`prompt_sha256` 校验**组合后的完整文本**。猫图哈希必须与该ID的已审阅描述一致。换图、改描述或动作后，需审阅并重建metadata；不匹配会报错，不回退到统一prompt。新prompt与锁帧设置均改变请求指纹，使用新的输出目录保存对照。旧metadata与旧动作prompt保留，无参数推理仍读取原成功案例。

需要调整时保留旧版，新建如 `cat_identity.v2.json` 并更新其中的 `prompt_version`；未实际推理验证继续标记 `pending_model_validation`。重建v1的命令如下；做v2时同时替换输入JSON和输出CSV文件名：

```bash
python3 scripts/prepare_ref2va_metadata.py \
  --identity-prompts exp_Ref2VA/prompts/cat_identity.v1.json --motion-ids '*' \
  --output exp_Ref2VA/metadata_identity_v1.csv \
  --smoke-output exp_Ref2VA/metadata_identity_v1_smoke.csv
```

`--motion-ids '*'`（保留引号）显式选择12个已登记动作。当前本地新增的 `Ref_lie_down.mp4`、`Ref_lie_down_mirror.mp4` 尚未登记独立动作prompt，不混入本轮对照；不指定动作筛选时，生成器仍会提醒未登记视频。向前走视频已同步当前名称 `Ref_walk_forward_to_screen.mp4`，旧表和新表的原视频/静音缓存路径一致；文件内容与旧版动作prompt不变，旧请求指纹不受该重命名影响。

本版只验证了输入、组合文本、请求协议和断点续跑。身份文字是生成条件，不能保证快速运动或遮挡中的ID保持；`--lock-first-frame` 还取决于服务端对混合关键帧条件的支持，真实效果仍待内网H100验证。

## 旧版统一动作 prompt：先预览，再批量生成

在内网工程根目录执行。沿用已跑通的 SGLang 环境和模型根目录；两个四卡副本共享任务队列，每个副本默认一个在途任务。

```bash
# 1. 检查全部960条输入；不访问服务、不创建缓存或请求预览
bash infer_ref2va_8gpu.sh --metadata exp_Ref2VA/metadata_all.csv --dry-run

# 2. 猫咪00逐个跑完12个参考视频，自动启动和关闭本次服务
bash infer_ref2va_8gpu.sh --start-servers \
  --model-path /srv/workspace/Kirin_AI_DataLake/models/MiniMax-H3 \
  --metadata exp_Ref2VA/metadata_smoke.csv \
  --output-dir outputs/ref2va_v1

# 3. 查看效果满意后，使用相同输出目录跑全量；跳过已完成的12条
bash infer_ref2va_8gpu.sh --start-servers \
  --model-path /srv/workspace/Kirin_AI_DataLake/models/MiniMax-H3 \
  --metadata exp_Ref2VA/metadata_all.csv \
  --output-dir outputs/ref2va_v1
```

预览结果位于 `outputs/ref2va_v1/videos/<motion_slug>/`，每个动作目录一条猫咪00视频。全量沿用相同文件名、seed、提示词和参数，正常情况下仅新增948条。中断后重跑相同命令即可继续。`--metadata` 可指定任意准备好的CSV，`--output-dir` 可指定绝对或相对保存目录。

若已有 Ref2VA 服务运行，去掉 `--start-servers` 即可复用默认30110/30112端口；也可用 `--server-urls http://127.0.0.1:30110 http://127.0.0.1:30112` 显式指定。需要连续调试时，终端A运行 `bash serve_ref2va_8gpu.sh --model-path /path/to/MiniMax-H3`，终端B运行上述推理命令并去掉 `--start-servers`，避免反复加载模型。脚本只管理自己启动的进程。

## 启用猫咪 ID 图首帧约束

客户端已实现 `--lock-first-frame` 的请求构造，但这不代表已跑通的内网服务支持该条件。2026-09-10内网实测返回 `task 'ref2va' does not allow condition role='keyframe' type='image'`，当前锁帧链路尚未跑通。该错误表示服务端任务规则拒绝Ref2VA混合关键帧；输入dry-run和本地模拟服务测试不能验证这项服务端能力。

在支持混合条件的服务上，同一猫图保留为 `image/reference`（对应 `<Picture 1>`），并额外作为 `image/keyframe, frame_index=0` 提交；静音动作视频仍为 `video/reference`（对应 `<Video 1>`）。预期约束作用于第0帧的整张图，包括猫咪姿态、构图和背景，并不是单独增加身份参考权重；所用权重的适用性、后续帧的ID保持和动作效果均需在H100上验证。

服务端兼容问题解决后，先用已跑通的历史 `drag_ear` 数据做一条对照：

```bash
# 只检查输入；不访问服务、不写请求预览
bash infer_ref2va_8gpu.sh --lock-first-frame \
  --metadata exp_Ref2VA/metadata.csv --limit 1 \
  --output-dir outputs/ref2va_lock_first_frame --dry-run

# 实际生成同一条，保留原seed及采样参数
bash infer_ref2va_8gpu.sh --lock-first-frame \
  --metadata exp_Ref2VA/metadata.csv --limit 1 \
  --output-dir outputs/ref2va_lock_first_frame
```

需要自动启动服务时，在实际生成命令中追加 `--start-servers --model-path /srv/workspace/Kirin_AI_DataLake/models/MiniMax-H3`。若原对照使用了 `--resolution` 或其他采样参数，锁帧实验保持相同设置。服务是否支持Ref2VA混合关键帧条件，以实际提交结果为准；客户端保留服务错误，不自动移除锁帧条件重试。

定位上述报错时，先收集实际服务环境的SGLang版本/源码commit和完整traceback，再决定是否回移混合条件支持。[上游任务规则](https://github.com/sgl-project/sglang/blob/v0.5.19/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/task_profiles.py)已有Ref2VA关键帧定义，但还需配套的请求校验、参考标签映射和关键帧条件传递，不能仅删除拒绝判断。暂不修改已跑通环境，不增加强制启动预检；只重启相同源码的服务或重复提交不会补齐能力。去掉参数可恢复普通Ref2VA，但不满足首帧约束目标；切换FL2VA也不能直接保留本请求的动作视频参考。

历史80猫批量去掉 `--limit 1` 即可，沿用同一目录可跳过已完成的锁帧任务。12动作预览与全量分别使用 `metadata_smoke.csv`、`metadata_all.csv`，另用 `outputs/ref2va_v1_lock_first_frame`，两次都保留 `--lock-first-frame`。开启或关闭锁帧会改变请求指纹，不能直接复用未锁帧的旧输出目录，否则会报参数不一致。无需用 `--force` 覆盖对照结果。

## 动作、参考视频和结束状态

所有目标请求均为4秒、24fps。逐个审阅了12个视频的3fps采样帧，并以6fps复核逗猫棒2的交叉前爪动作。镜像方向指**画面左右**；猫图本身不翻转，保留不对称花纹和自然耳尾特征。玩具作为独立参考主体保留，无可见操作者。

| motion_slug / 单动作CSV名 | 原始参考视频 | 主要动作与结束状态 | 条数 |
| --- | --- | --- | --- |
| `drag_ear` | `Ref_drag_ear.mp4` | 自己的左前爪（画面右）擦耳脸后恢复坐姿；原成功对照 | 80 |
| `drag_ear_mirror` | `Ref_drag_ear_mirror.mp4` | 自己的右前爪（画面左）擦耳脸后恢复坐姿 | 80 |
| `cat_teaser1` | `Ref_cat_teaser1.mp4` | 坐姿左前爪拨打画面右侧羽毛，落爪后继续注视 | 80 |
| `cat_teaser1_mirror` | `Ref_cat_teaser1_mirror.mp4` | 坐姿右前爪拨打画面左侧羽毛，落爪后继续注视 | 80 |
| `cat_teaser2` | `Ref_cat_teaser2.mp4` | 起身，右前爪跨胸向画面右侧追拍，小步移位后站定 | 80 |
| `cat_teaser2_mirror` | `Ref_cat_teaser2_mirror.mp4` | 起身，左前爪跨胸向画面左侧追拍，小步移位后站定 | 80 |
| `cat_teaser3` | `Ref_cat_teaser3.mp4` | 双前爪抬起抓扑、随玩具下探落地，最后坐定 | 80 |
| `drag_yarn_ball_and_catch` | `Ref_drag_yarn_ball_and_catch.mp4` | 球由画面左向右滚动，猫前扑并用双爪按住，站立俯身结束 | 80 |
| `drag_yarn_ball_and_catch_mirror` | `Ref_drag_yarn_ball_and_catch_mirror.mp4` | 球由画面右向左滚动，猫前扑并用双爪按住，站立俯身结束 | 80 |
| `curl_up_and_lie_down` | `Ref_curl_up_and_lie_down.mp4` | 低头屈腿蜷卧，头在画面左侧，闭眼休息 | 80 |
| `curl_up_and_lie_down_mirror` | `Ref_curl_up_and_lie_down_mirror.mp4` | 低头屈腿蜷卧，头在画面右侧，闭眼休息 | 80 |
| `walk_forward_to_screen` | `Ref_walk_forward_to_screen.mp4` | 起身交替迈步靠近固定镜头，放大为近处站姿并略偏头 | 80 |

例如只生成起身追拍动作的80只猫：

```bash
bash infer_ref2va_8gpu.sh --metadata exp_Ref2VA/metadata/cat_teaser2.csv \
  --output-dir outputs/ref2va_v1
```

多数参考的动作在前约3秒完成，后段保持各自结束状态；提示词保留这一时序。向前走参考的视频轨长4.041699秒，metadata保留实测参考时长，目标仍请求4秒。模型帧数按17n+5对齐，实际输出可能长于4秒，保持模型原始输出。

## 需要同步的文件

代码通过 `zonghui-liu-opt/MiniMax-H3` 的 `data_pipeline` 分支同步。输入使用相对CSV所在位置的路径，可整目录搬运，不依赖本地Mac绝对路径。需在内网保留：

- `data_h3/cat_ids/` 原有80张猫图；`cat_catalog.csv` 为ID与文件名映射。
- 本目录上述12个原始MP4均已纳入Git，随 `data_pipeline` 分支一起同步，文件名与表格一致。蜷卧原视频统一命名为 `Ref_curl_up_and_lie_down.mp4`。
- `motions.json`、`prompts/*.v1.en.txt`、`metadata_all.csv`、`metadata_smoke.csv` 和 `metadata/*.csv`。
- 本次逐猫身份实验另需 `prompts/cat_identity.v1.json`、`metadata_identity_v1.csv` 和 `metadata_identity_v1_smoke.csv`。

只维护一份推理实现、一份metadata生成器，以及按版本保存的动作提示词。metadata引用提示词文件并记录提示词/源视频哈希，不在960行里重复嵌入正文。静音参考、日志、state、请求预览、生成视频和清单均为忽略的运行产物；无需同步工程包、上游源码快照或静音副本。

蜷卧视频去除文件名前导空格后，原视频与静音缓存路径已在smoke、全量及单动作metadata中同步更新。新缓存 `prepared/Ref_curl_up_and_lie_down.motion.mp4` 在实际推理时自动准备。视频内容、动作提示词、任务编号和请求指纹保持不变，已有输出目录可以继续使用，已完成结果仍按原规则跳过。

历史 `metadata.csv`（80条 `drag_ear`、seed 0、旧编号）及 `drag_ear.v1.en.txt` 保持原样。无参数的 `infer_ref2va_8gpu.sh` 仍沿用该成功案例；新任务请显式传入 `metadata_smoke.csv` 或 `metadata_all.csv`。新增全量网格的编号与历史网格不同，使用上述新输出目录。预览、按动作拆分和全量CSV之间的编号完全一致。

## 更新提示词、换预览猫或重建metadata

新增11份prompt均标记 `pending_model_validation`。先看预览中猫咪身份、前爪左右、接触关系、玩具轨迹及最后姿态，再决定是否改参考视频或提示词。旧提示词按版本保留；修改动作时新建如 `cat_teaser2.v2.en.txt`，更新 `motions.json` 的 `prompt_file`、`prompt_version` 和观察信息后重建：

```bash
# 一次生成全量、固定猫咪00的预览，以及每动作80条CSV
python3 scripts/prepare_ref2va_metadata.py \
  --output exp_Ref2VA/metadata_all.csv \
  --smoke-output exp_Ref2VA/metadata_smoke.csv \
  --per-motion-dir exp_Ref2VA/metadata

# 换成猫咪38检查全部动作；默认seed仍为0
python3 scripts/prepare_ref2va_metadata.py \
  --smoke-cat-id 38 --smoke-output exp_Ref2VA/metadata_smoke.csv

# 只准备选定动作和猫；种子数量仅在显式要求时增加
python3 scripts/prepare_ref2va_metadata.py --cat-ids 00 02 38 \
  --motion-ids cat_teaser2 cat_teaser2_mirror --seeds 0 10000 20000 \
  --output exp_Ref2VA/metadata_subset.csv
```

生成器未指定 `--identity-prompts` 时默认输出 `metadata_all.csv`，启用身份配置时默认输出 `metadata_identity_v1.csv`；默认只有seed 0，不覆盖历史 `metadata.csv`。`--smoke-output` 从当前完整任务网格选取一只猫，`--per-motion-dir` 按动作拆分，保留任务名并自动重定位相对路径。`--cat-ids`、`--motion-ids` 支持数字编号或完整ID，动作也支持slug；`--smoke-cat-id` 必须属于本次所选猫。推理 `--limit` 取筛选后的前N条，不等于覆盖所有动作；预览可直接使用相应版本的12条smoke表。

同一动作配置和seed列表下，筛选不会改变任务编号。增减注册动作或seed列表会改变任务网格编号，需要同步重建对应的预览与拆分CSV并使用新输出目录。改变提示词、视频、分辨率或采样参数时也建议使用如 `outputs/ref2va_v2` 的新目录，保留前后效果对照。

## 沿用现有推理环境

模型入口固定 `--model-variant ref2va`。`--model-path` 传含完整 `Ref2VA/` 分区的模型根目录，不能传 `Ref2VA` 子目录或FL2VA分区。入口保持Hub离线模式；不添加包版本、源码能力、权重或编码器的强制启动预检，不修改已跑通的环境。

| 副本 | GPU | HTTP | ZMQ | master | scheduler |
| --- | --- | --- | --- | --- | --- |
| 1 | 0,1,2,3 | 30110 | 30111 | 31110 | 32110 |
| 2 | 4,5,6,7 | 30112 | 30113 | 31111 | 32111 |

部署参数仍使用 `--gpu-groups`、`--tp-size`、`--ulysses-degree` 和原端口选项。客户端仅使用Python标准库及现有 `ffmpeg`、`ffprobe`。实际推理首次运行时用 `-c:v copy -an` 自动生成 `prepared/` 静音缓存，不重新编码；之后校验哈希复用。`--dry-run` 仅检查输入，只有显式 `--write-requests` 才输出调试请求预览。生成metadata时也默认不生成媒体，需要提前准备可加 `--prepare-media`。

请求继续为 `task=ref2va`，按顺序提交猫图 `image/reference` 和静音视频 `video/reference`，对应 `<Picture 1>` 身份/环境与 `<Video 1>` 动作/玩具/时序。新增提示词使用 `[reference generation]`，不要求返回首帧。默认无首尾帧硬锁；`--lock-first-frame` 仅在显式要求时额外提交同图 `image/keyframe, frame_index=0`，服务不支持会保留实际错误，不自动降级。去除源视频音轨避免自动引入音频参考。

默认50步、768短边、9:16、flow_shift=12、audio_flow_shift=3。新预览先沿用已成功的设置，方便判断参考和prompt本身的效果。

## 可选原生低分辨率

原有 `--resolution WIDTHxHEIGHT` 功能继续可用，例如480×832；不做生成后缩放。宽高须为32的倍数，并能由H3支持的比例及画布规则精确推导。先在原服务终端停止旧服务，再显式开启自定义分辨率：

```bash
# 终端A：一次启动，之后预览和批量共用
bash serve_ref2va_8gpu.sh --resolution 480x832

# 终端B：预览；切全量时保持分辨率与输出目录一致
bash infer_ref2va_8gpu.sh --resolution 480x832 \
  --metadata exp_Ref2VA/metadata_smoke.csv --output-dir outputs/ref2va_v1_480x832
```

也可将 `--start-servers --resolution 480x832` 加入一体化推理命令。480×832发送 `short_edge=468, aspect_ratio=9:16`，原生得到精确画布；直接指定480短边会对齐为480×864。已存在的内存兼容处理仅在显式指定分辨率并启动服务时放开旧版两处短边768限制，在主进程和GPU工作进程生效，不写入SGLang安装目录或字节码，不检查版本。保留比例、32像素对齐和像素上限规则。

分辨率进入请求指纹，下载后也核实MP4实际宽高；不匹配不会标记完成。低分辨率的速度和画质仍需实际GPU验证。接口依据：[SGLang H3文档](https://docs.sglang.io/cookbook/diffusion/MiniMax/MiniMax-H3)、[原画布规则](https://github.com/sgl-project/sglang/blob/v0.5.19/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/resolved_plan.py)。

## 恢复与验证

同请求跳过已下载视频；未完成ID回原服务恢复；服务重启后ID返回404才重提。输入或采样参数变化会拒绝复用同名旧任务。更换模型权重需用新输出目录，因为不会哈希整套权重。

`--retry-failed` 重提明确失败任务；轮询超时保留ID。POST响应丢失标记 `submission_unknown`，先检查服务日志，再决定是否重跑。`--force` 先把旧结果归档至忽略的 `history/` 再重跑。输出目录有进程锁。成功退出0、部分失败1、配置错误2、中断130。

本地验证命令：`python3 -m unittest discover -s tests -q`。测试覆盖metadata拆分和路径搬运、默认单seed、非法输入保护、默认参考请求、双副本调度、单猫多动作预览到全量的结果复用，以及既有恢复和失败重试。输入检查和模拟服务测试不代表真实H100的生成质量；本次没有在内网执行模型推理。
