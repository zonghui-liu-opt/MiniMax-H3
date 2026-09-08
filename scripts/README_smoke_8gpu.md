# smoke v3 首帧引导的 8 × H100 并行推理

从仓库根目录执行；使用已经跑通 `infer.sh` 的 SGLang 环境，无需安装新依赖。
默认仅处理 `data_h3/metadata_smoke_v3.csv` 的 36 条数据，以每行的 `input_image`
作为唯一首帧条件，prompt 按 CSV 原文发送。图片相对路径以 CSV 所在目录解析。

```bash
# 校验 v3 的全部 36 条输入、打印部署命令、生成请求预览；不启动模型
bash infer_smoke_8gpu.sh --dry-run

# 自动启动两个四卡服务，等待加载完成，再跑完 v3
bash infer_smoke_8gpu.sh \
  --model-path /srv/workspace/Kirin_AI_DataLake/models/MiniMax-H3
```

首帧模式（I2VA）在 SGLang 中仍使用 `--model-variant fl2va` 和 `"task": "fl2va"`。
每条请求的 `conditions` 只有一张 `role: keyframe`、`frame_index: 0` 的图片，
不添加 `frame_index: -1`，也不读取 CSV 的 `last_image` 作为条件。
这是 [SGLang 官方支持的首帧模式](https://docs.sglang.io/cookbook/diffusion/MiniMax/MiniMax-H3#4-generate-video-and-audio)，
与仓库 `scripts/readme/full-2k-i2va-h3-base.sh` 的条件结构一致。

默认 GPU 分组 `0,1,2,3` 和 `4,5,6,7`，每组沿用已跑通的
`--num-gpus 4 --tp-size 2 --ulysses-degree 2 --performance-mode speed`。
默认端口分配如下：

| GPU | HTTP | ZMQ Broker（自动占用 HTTP+1） | master | scheduler |
| --- | --- | --- | --- | --- |
| 0,1,2,3 | 30010 | 30011 | 31010 | 32010 |
| 4,5,6,7 | 30012 | 30013 | 31011 | 32011 |

`--base-port` 设置第一个 HTTP 端口，后续副本每次加 **2**。
启动前校验全部八个端口的范围、重复和实际占用，包括隐式的 ZMQ Broker 端口。
命令、客户端 URL 和日志名使用同一份端口分配。

旧脚本使用 HTTP `30010/30011`，但第一个服务的 ZMQ Broker 会占用 `30011`，
导致第二个服务在模型加载完成后报 `address already in use`。此次修复为每个副本
预留 HTTP+1；不要再把第二个 HTTP 端口手动设为 `30011`。
SGLang Diffusion 的两个副本如果都使用默认 `master_port=30005`，可能在同时启动时
发生 `EADDRINUSE`；仅分开 HTTP 端口不能隔离分布式初始化。
本入口不传入 `--nccl-port`。旧参数 `--base-nccl-port` 作为 `--base-master-port` 的兼容别名，
实际始终传入 Diffusion 使用的 `--master-port`。
各组内部做模型并行，两组之间处理不同视频。v3 的全部任务共用一个队列，
谁先完成谁领取下一条；不会给慢副本固定分配一半数据。
每个副本默认仅一个在途请求，总计两个，避免把 HTTP 排队误当成 GPU 并行。

[SGLang 官方 H3 配置](https://docs.sglang.io/cookbook/diffusion/MiniMax/MiniMax-H3)
验证了四张 H100 80 GB 上的 TP2 + Ulysses2 配置。本入口以这个配置作为默认，
不宣称它已在你的机器上达到最大吞吐；主机内存、磁盘加载速度、GPU 互联和实际输入都会影响表现。
启动两份完整模型也会增加主机内存与磁盘带宽需求。可先运行 `nvidia-smi topo -m`，
通过 `--gpu-groups` 把互联更好的四张卡放入同组。

## 已有服务与参数调整

如果已经运行了原来的四卡服务（HTTP 30010、master 30005），请先停止它再使用自动启动入口，
或者在其余四张卡上启动第二个服务，并显式使用不同的内部端口：

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 sglang serve \
  --model-path /srv/workspace/Kirin_AI_DataLake/models/MiniMax-H3 \
  --num-gpus 4 --tp-size 2 --ulysses-degree 2 \
  --performance-mode speed --host 127.0.0.1 \
  --port 30012 --master-port 31011 --scheduler-port 32011 --model-variant fl2va

# 在另一个终端连接两组服务；该模式不会启动或停止它们
bash infer_smoke_8gpu.sh \
  --server-urls http://127.0.0.1:30010 http://127.0.0.1:30012
```

两个服务必须使用不同 GPU，不能把同一服务的两个 URL 当作两个副本。
启动器不会自动接管占用端口的服务；端口冲突会在模型启动前报出副本、端口用途和号码。
`--server-urls` 模式使用你提供的地址，不会自动改写已有服务地址。
自动启动模式会在完成、失败或收到 Ctrl-C/SIGTERM 后清理本次启动的服务进程组。
可在 tmux 中运行完整命令以保持长任务。

继承原客户端的采样参数：50 步、768 短边、4 秒、seed 从 0 按行递增。
v3 提示词为四秒视频，与默认时长一致；文件名带原始行索引。
默认输出改为 `results/smoke_v3_i2va`，避免复用旧版首尾帧结果。
参数会写入请求预览。修改 prompt、时长、首尾帧模式或采样设置后应使用新输出目录，
因为断点逻辑沿用原客户端的文件名匹配，不会自动判断配置是否变化。

```bash
# 小规模检查：仅 v3 前 2 条，共 2 条
bash infer_smoke_8gpu.sh --limit 2 --output-dir results/smoke_v3_i2va_check

# 在默认并发 1 跑通后，对比每副本 2 个在途请求是否能提高实际吞吐
bash infer_smoke_8gpu.sh --max-concurrency 2 --output-dir results/smoke_v3_i2va_c2
```

`--metadata` 可替换默认 CSV；默认不会读取 v1/v2，也不会读取
`metadata_smoke_v3_i2va.csv` 或 `metadata_smoke_v3_fl2va.csv`。
`--single-frame` 已默认开启，无需额外传入。
保留旧模式的显式入口：`--first-last-frame` 启用首尾帧引导，
`--metadata-v1` 仅在显式提供时追加第二份 CSV。
如需重现旧版两份 CSV 的推理，可执行：

```bash
bash infer_smoke_8gpu.sh --first-last-frame \
  --metadata data_h3/metadata_smoke_v2.csv \
  --metadata-v1 data_h3/metadata_smoke_v1.csv \
  --output-dir results/smoke
```

`--max-concurrency` 控制每个副本的客户端在途请求数，不保证 SGLang 同时执行多个视频。
不默认降低步数、切换权重或启用近似缓存。`--file-uri` 可减少图片上传，
仅适用于客户端与服务端能读取相同绝对路径的情况，默认仍发送 data URI。
`--gpu-groups`、`--tp-size`、`--ulysses-degree` 可用于实机拓扑对比；
其他卡数/拓扑必须先验证显存容量与兼容性。额外服务参数按 token 传入，例如
`--server-arg=--encoder-parallel --server-arg=auto`。

## 输出与恢复

```text
results/smoke_v3_i2va/
  manifest.json                  # 总量、耗时、新提交并完成的吞吐（不含模型启动）
  logs/<本次运行编号>/server_30010.log
  logs/<本次运行编号>/server_30012.log
  metadata_smoke_v3/              # 使用 CSV 文件名（不含扩展名）作为子目录
    manifest.json
    videos/*.mp4
    states/*.json
    requests/*.json
```

重复相同命令会跳过已有 MP4，未完成任务按状态中的 `server_url` 回到原服务恢复。
服务重启后如果旧 ID 返回 404，则重新提交该条。明确的服务端失败默认保留，
使用 `--retry-failed` 才重提；HTTP 暂时失败或轮询超时仍保留 ID 以继续恢复。
`--force` 强制重跑全部选中数据。若原服务地址改变，先恢复原地址，或确认重跑后用 `--force`。
旧客户端没有记录服务地址的状态文件按 `--server-url` 归属恢复。
输出目录有进程锁，防止两个新入口同时覆盖相同结果。
每次启动服务都会创建独立的日志目录，并打印本次日志的完整路径；旧日志保留供排查，
不会继续追加到旧文件。日志首行记录实际启动命令和 GPU 分组。
服务启动失败或超时时，终端自动打印各服务日志末尾（最多 40 行 / 8 KiB），
并清理本次启动的进程组。看到第一个服务也退出时，应先看第二个服务的原始错误，
这可能是启动器发现某个副本失败后触发的统一清理。

所有视频成功退出码为 0，部分失败为 1，配置/启动错误为 2，中断为 130。
中断时停止领取新任务；正在进行的 HTTP 调用最多等待本次请求超时后退出。
结果按 MP4 文件头验证，沿用旧客户端；这不代替实际视频内容和音画质量检查。

本地验证：`python3 -m unittest discover -s tests -v`。
回归测试用真实子进程绑定 HTTP、HTTP+1 Broker、master、scheduler 四类端口，
覆盖旧布局冲突、全部八个端口的占用预检、端口越界、双副本提交和恢复、
启动失败/超时诊断与进程清理，以及默认只读取 v3、双副本仅发送首帧条件、
忽略尾帧列、下载和恢复、显式首尾帧兼容模式；不需要安装 SGLang 或访问 GPU。
这些校验不代表已完成真实 GPU 推理，视频生成效果仍需实机验证。
