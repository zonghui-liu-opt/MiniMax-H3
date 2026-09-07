# 两份 smoke CSV 的 8 × H100 并行推理

从仓库根目录执行；使用已经跑通 `infer.sh` 的 SGLang 环境，无需安装新依赖。

```bash
# 校验全部 72 条输入、打印部署命令、生成请求预览；不启动模型
bash infer_smoke_8gpu.sh --dry-run

# 自动启动两个四卡服务，等待加载完成，再跑完两份 CSV
bash infer_smoke_8gpu.sh \
  --model-path /srv/workspace/Kirin_AI_DataLake/models/MiniMax-H3
```

默认 GPU 分组 `0,1,2,3` 和 `4,5,6,7`，每组沿用已跑通的
`--num-gpus 4 --tp-size 2 --ulysses-degree 2 --performance-mode speed`。
HTTP 端口为 `30010/30011`，通信端口沿用当前 SGLang 版本自身的配置。
默认不传入 `--nccl-port`，以兼容不提供该命令行参数的 SGLang 版本。
如果已确认当前版本支持该参数，可以显式使用 `--base-nccl-port 31010`，
为两个副本分别传入 `31010/31011`，并启用对应端口的占用检查。
各组内部做模型并行，两组之间处理不同视频。两份 CSV 共用任务队列，
谁先完成谁领取下一条；不会给慢副本固定分配一半数据。
每个副本默认仅一个在途请求，总计两个，避免把 HTTP 排队误当成 GPU 并行。

[SGLang 官方 H3 配置](https://docs.sglang.io/cookbook/diffusion/MiniMax/MiniMax-H3)
验证了四张 H100 80 GB 上的 TP2 + Ulysses2 配置。本入口以这个配置作为默认，
不宣称它已在你的机器上达到最大吞吐；主机内存、磁盘加载速度、GPU 互联和实际输入都会影响表现。
启动两份完整模型也会增加主机内存与磁盘带宽需求。可先运行 `nvidia-smi topo -m`，
通过 `--gpu-groups` 把互联更好的四张卡放入同组。

## 已有服务与参数调整

如果已经运行了原来的四卡服务，请先停止它再使用自动启动入口，或者在其余四张卡上启动第二个服务：

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 sglang serve \
  --model-path /srv/workspace/Kirin_AI_DataLake/models/MiniMax-H3 \
  --num-gpus 4 --tp-size 2 --ulysses-degree 2 \
  --performance-mode speed --host 127.0.0.1 \
  --port 30011 --model-variant fl2va

# 在另一个终端连接两组服务；该模式不会启动或停止它们
bash infer_smoke_8gpu.sh \
  --server-urls http://127.0.0.1:30010 http://127.0.0.1:30011
```

两个服务必须使用不同 GPU，不能把同一服务的两个 URL 当作两个副本。
启动器不会自动接管占用端口的服务；端口冲突会直接报错。
自动启动模式会在完成、失败或收到 Ctrl-C/SIGTERM 后清理本次启动的服务进程组。
可在 tmux 中运行完整命令以保持长任务。

继承原客户端的采样参数：50 步、768 短边、4 秒、同图首尾帧、seed 从 0 按行递增。
两份 CSV 各自从 seed 0 开始，方便对照 v1/v2；每份的文件名仍带原始行索引。
提示词中的末帧时间为 4.46 秒；需要对齐时显式使用 `--duration-seconds 4.46`。
参数会写入请求预览。修改 prompt、时长、采样设置后应使用新输出目录，
因为断点逻辑沿用原客户端的文件名匹配，不会自动判断配置是否变化。

```bash
# 小规模检查：每份 CSV 前 2 条，共 4 条
bash infer_smoke_8gpu.sh --limit 2 --output-dir results/smoke_check

# 与提示词时长对齐
bash infer_smoke_8gpu.sh --duration-seconds 4.46 --output-dir results/smoke_4_46s

# 在默认并发 1 跑通后，对比每副本 2 个在途请求是否能提高实际吞吐
bash infer_smoke_8gpu.sh --max-concurrency 2 --output-dir results/smoke_c2
```

`--max-concurrency` 控制每个副本的客户端在途请求数，不保证 SGLang 同时执行多个视频。
不默认降低步数、切换权重或启用近似缓存。`--file-uri` 可减少图片上传，
仅适用于客户端与服务端能读取相同绝对路径的情况，默认仍发送 data URI。
`--gpu-groups`、`--tp-size`、`--ulysses-degree` 可用于实机拓扑对比；
其他卡数/拓扑必须先验证显存容量与兼容性。额外服务参数按 token 传入，例如
`--server-arg=--encoder-parallel --server-arg=auto`。

## 输出与恢复

```text
results/smoke/
  manifest.json                  # 总量、耗时、新提交并完成的吞吐（不含模型启动）
  logs/server_30010.log
  logs/server_30011.log
  metadata_smoke_v2/
    manifest.json
    videos/*.mp4
    states/*.json
    requests/*.json
  metadata_smoke_v1/
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

所有视频成功退出码为 0，部分失败为 1，配置/启动错误为 2，中断为 130。
中断时停止领取新任务；正在进行的 HTTP 调用最多等待本次请求超时后退出。
结果按 MP4 文件头验证，沿用旧客户端；这不代替实际视频内容和音画质量检查。

本地验证：`python3 -m unittest discover -s tests -v`。
