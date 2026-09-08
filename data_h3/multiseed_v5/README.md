# Smoke V5：可读文件名与多 seed 推理

已按用户确认，将跳跃（13）和玩逗猫棒（14）加入 Smoke V5。保留 V4 的 12 个动作和原有 3 张猫图。英文 prompt 与已审核稿逐字一致，实际生成效果待内网推理验证。

| 文件 | 猫咪/动作组合 | 每个组合的 seed | 推理条数 |
| --- | ---: | --- | ---: |
| `data_h3/metadata_smoke_v5.csv` | 3 × 14 = 42 | 0 | 42 |
| `data_h3/metadata_smoke_v5_multiseed.csv` | 3 × 14 = 42 | 0、10000、20000 | 126 |

同一个猫咪/动作组合连续排列多个 seed，图片和 prompt 完全相同。各组合使用同一组明确的 seed；seed 不再叠加源 CSV 行号。CSV 中的 `seed` 是请求实际使用的值，优先于推理命令的 `--seed`。

本次先生成 Smoke V5；原有 V4 CSV 和原始测试集保留。全量 80 张猫图的新增动作仍保存在上一轮待追加稿中。

## 文件名

```text
id_cat-id_motion-id_seed-id.mp4

036_00-orange-shorthair-mackerel-tabby_13-small-jump_seed-0.mp4
037_00-orange-shorthair-mackerel-tabby_13-small-jump_seed-10000.mp4
038_00-orange-shorthair-mackerel-tabby_13-small-jump_seed-20000.mp4
039_00-orange-shorthair-mackerel-tabby_14-play-teaser-wand_seed-0.mp4
```

- `id`：当前推理 CSV 的零起始行号，至少补齐 3 位，由现有客户端添加。筛选生成另一份 CSV 后从 000 重新编号。
- `cat-id`：猫编号加可读英文名称，例如 `00-orange-shorthair-mackerel-tabby`。
- `motion-id`：动作编号加可读英文名称，例如 `13-small-jump`。
- `seed-id`：`seed-` 加实际随机种子值，例如 `seed-10000`。

CSV 的 `output_name` 只包含后三段，既不包含前面的行号，也不带 `.mp4`。现有推理客户端会自动补齐。英文名称使用连字符连接，四个字段之间用下划线分隔；因此不会被客户端的 ASCII 文件名清理规则去掉。

扩展 CSV 额外保留 `source_id`，指向基础 V5 CSV 的零起始行号。同一组合的所有 seed 具有同一个 `source_id`。`cat_idx`、`motion_idx` 保留简短编号，`cat_name_zh`、`motion_name_zh` 提供中文对照。

## 直接推理已准备的 126 条

在内网仓库根目录运行，复用已经启动的 SGLang 服务：

`infer_smoke_8gpu.sh` 的默认输入现已切换为本节的 V5 多 seed CSV；直接运行也会使用 126 条 V5 请求。显式传入 `--metadata` 时，以指定文件为准。

```bash
bash infer_smoke_8gpu.sh \
  --metadata data_h3/metadata_smoke_v5_multiseed.csv \
  --single-frame --duration-seconds 4 \
  --output-dir results/v5_multiseed_smoke
```

视频保存到 `results/v5_multiseed_smoke/metadata_smoke_v5_multiseed/videos/`。

默认连接本机 30010 和 30012 两个服务。使用其他服务地址时，增加 `--server-urls` 并填写实际 URL。加 `--dry-run` 可只生成请求预览。这里仍是 4 秒首帧引导，不添加尾帧条件，也不自动裁剪视频。

启动日志会打印实际 CSV 绝对路径、seed 来源、视频目录和前三个输出文件名。若仍看到 `000_00_01_0.mp4`，说明正在使用旧编号格式，或查看的是旧输出目录；核对日志中的实际路径。旧 V4 多 seed CSV 第一轮使用源行号作为 seed，因此会出现 seed 与 id 相等。

内网需要数字图片文件名时，使用 [V5 ASCII 数据包](../h100_v5_ascii/README.md)，将 `--metadata` 改为 `data_h3/h100_v5_ascii/metadata_smoke_v5_multiseed.csv`。`h100_v4_ascii` 目录保留旧 V4 数据，不能通过运行新脚本自动变成 V5。

## 指定同一只猫、指定动作和多个 seed

下面先选短毛橘猫（00）的跳跃（13）与玩逗猫棒（14），每个动作使用三个 seed，共 6 条：

```bash
python3 scripts/prepare_multiseed_metadata.py \
  --metadata data_h3/metadata_smoke_v5.csv \
  --cat-ids 00 \
  --motion-ids 13 14 \
  --seeds 0 10000 20000 \
  --output data_h3/metadata_smoke_v5_selected.csv

bash infer_smoke_8gpu.sh \
  --metadata data_h3/metadata_smoke_v5_selected.csv \
  --single-frame --duration-seconds 4 \
  --output-dir results/v5_selected
```

只测跳跃时，将 `--motion-ids 13 14` 改为 `--motion-ids 13`。需要其他随机种子时直接替换 `--seeds`，例如 `--seeds 42 43 44 45`。`--cat-ids` 和 `--motion-ids` 都支持多个编号，也支持完整英文 ID。省略某个筛选参数会保留该维度的全部条目。

生成器的输入应使用基础 V5 CSV，不要再次输入已扩展的多 seed CSV。输出 CSV 可放在其他目录，图片相对路径会自动重算。内网只运行已准备的多 seed CSV 时，无需修改现有推理客户端；自定义筛选或 seed 时再同步新增的 `scripts/prepare_multiseed_metadata.py`。

## 编号对照

| cat_idx | cat-id | 猫咪 |
| --- | --- | --- |
| 00 | 00-orange-shorthair-mackerel-tabby | 短毛橘猫鱼骨纹 |
| 02 | 02-orange-longhair | 长毛橘猫 |
| 38 | 38-peterbald | 彼得秃猫 |

| motion_idx | motion-id | 动作 |
| --- | --- | --- |
| 01 | 01-idle-blink | 自然呼吸与眨眼 |
| 02 | 02-left-ear-tug | 左耳轻拽回弹 |
| 03 | 03-right-ear-tug | 右耳轻拽回弹 |
| 04 | 04-right-paw-yarn-left-to-right | 右前爪拨球，画面左到右 |
| 05 | 05-left-paw-yarn-right-to-left | 左前爪拨球，画面右到左 |
| 06 | 06-roll-left-belly-up-swat | 向左打滚、露腹挥爪 |
| 07 | 07-roll-right-belly-up-swat | 向右打滚、露腹挥爪 |
| 08 | 08-lick-left-paw | 舔左前爪 |
| 09 | 09-lick-right-paw | 舔右前爪 |
| 10 | 10-eat-cat-food | 吃猫粮 |
| 11 | 11-curl-left-sleep | 向左蜷卧入睡 |
| 12 | 12-curl-right-sleep | 向右蜷卧入睡 |
| 13 | 13-small-jump | 小幅跳跃 |
| 14 | 14-play-teaser-wand | 玩逗猫棒 |

左右肢体和打滚方向均以猫自身为准，球的移动方向按画面标注。

验证记录见同目录 `validation.json`；已确认数据覆盖、请求 seed、文件名和首帧条件，未执行真实 GPU 推理。
