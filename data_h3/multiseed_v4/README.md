# V4 多 seed 数据

复制 V4 的图片与 prompt，仅扩展 seed。没有改写任何 prompt，也没有执行模型推理。

| 文件 | 原始条数 | 每条 seed 数 | 扩展条数 |
| --- | ---: | ---: | ---: |
| `data_h3/metadata_smoke_v4_multiseed.csv` | 36 | 3 | 108 |
| `data_h3/metadata_v4_multiseed.csv` | 960 | 3 | 2,880 |

## Seed 与行序

设 `i` 为对应原始 V4 CSV 的零起始行号，三个 seed 为 `i`、`10000 + i`、`20000 + i`。
第一轮与原始 V4 在默认 `--seed 0 --seed-stride 1` 下的 seed 相同。
文件按轮次排列，每轮保持对应原始 CSV 的完整行序。

Smoke 和全量各自以自己的源 CSV 行号计算 seed，保留各自默认基线；同一只猫、同一动作在两份文件中的源行号不一定相同。
CSV 已填写显式 `seed` 列，客户端会优先读取它，无需另传 `--seed`。

## 输出文件名

格式：`id_cat-id_motion-id_seed.mp4`。

| 部分 | 定义 |
| --- | --- |
| `id` | 当前扩展 CSV 的零起始行号，至少补齐 3 位，例如 `000`、`960`、`1920` |
| `cat-id` | 输入图片文件名开头的猫编号，`00`–`79` |
| `motion-id` | 既定动作编号，`01`–`12`，见下表 |
| `seed` | 该条请求实际使用的整数 seed |

全量文件中同一只猫、同一动作的三个例子：

```text
000_00_01_0.mp4
960_00_01_10000.mp4
1920_00_01_20000.mp4
```

全量最后一条为 `2879_79_12_20959.mp4`。

新增的 `id`、`cat_id`、`motion_id` 列用于索引和检查；现有客户端自动以实际 CSV 行号生成 `id` 前缀。
`output_name` 列只填写 `cat-id_motion-id_seed`，例如 `00_01_10000`。
客户端自动添加行号前缀和 `.mp4`，因此不能在 `output_name` 中再次加入 id 或扩展名。
保持 CSV 行序即可复现本文件约定的 id；更改行序会改变客户端生成的 id 前缀。

| motion-id | 动作 |
| --- | --- |
| 01 | 自然呼吸＋自然眨眼 |
| 02 | 左耳轻拽回弹 |
| 03 | 右耳轻拽回弹 |
| 04 | 右前爪拨毛线球，画面左→右 |
| 05 | 左前爪拨毛线球，画面右→左 |
| 06 | 向左打滚、露腹挥爪 |
| 07 | 向右打滚、露腹挥爪 |
| 08 | 舔左前爪 |
| 09 | 舔右前爪 |
| 10 | 吃一口猫粮 |
| 11 | 向左收身蜷卧入睡 |
| 12 | 向右收身蜷卧入睡 |

左右以猫自身为准，球的移动方向单独以画面方向标明。

## 运行

在仓库根目录执行，复用已经启动的两组 SGLang 服务：

```bash
# Smoke：108 条
bash infer_smoke_8gpu.sh \
  --metadata data_h3/metadata_smoke_v4_multiseed.csv \
  --single-frame --duration-seconds 4 \
  --output-dir results/v4_multiseed_smoke

# 全量：2,880 条
bash infer_smoke_8gpu.sh \
  --metadata data_h3/metadata_v4_multiseed.csv \
  --single-frame --duration-seconds 4 \
  --output-dir results/v4_multiseed_full
```

输出位置分别为：

```text
results/v4_multiseed_smoke/metadata_smoke_v4_multiseed/videos/
results/v4_multiseed_full/metadata_v4_multiseed/videos/
```

若只连接一组现有服务，可追加 `--server-urls http://127.0.0.1:30010`，或使用你的实际服务地址。
在命令末尾追加 `--dry-run`，即可只生成请求预览而不访问模型服务。

仍按 V4 的四秒请求、前三秒动作设计运行。此处没有裁剪视频；保留前 73 帧的处理仍在后续执行。
源文件与生成文件的 SHA256、数据规模、命名规则记录在同目录 `manifest.json`。
