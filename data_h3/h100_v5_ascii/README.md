# Smoke V5：内网 ASCII 路径数据包

本目录包含 3 张数字文件名的猫图、42 条基础 V5 数据和 126 条 V5 多 seed 数据。
图片字节、prompt、seed 和可读输出名与 `data_h3/metadata_smoke_v5*.csv` 一致；仅修改输入图片相对路径。

在仓库根目录运行：

```bash
bash infer_smoke_8gpu.sh \
  --metadata data_h3/h100_v5_ascii/metadata_smoke_v5_multiseed.csv \
  --output-dir results/v5_ascii_multiseed
```

需要复制到内网时，复制整个目录，保留内部 `cat_ids/00.png`、`02.png`、`38.png` 的结构。
上面的 `--metadata` 必须指向本目录的 V5 CSV；旧 `h100_v4_ascii` 保留旧 V4 的编号和 seed 规则。

提交前日志应显示 126 条请求；前三条属于同一猫咪和动作：

```text
000_00-orange-shorthair-mackerel-tabby_01-idle-blink_seed-0.mp4
001_00-orange-shorthair-mackerel-tabby_01-idle-blink_seed-10000.mp4
002_00-orange-shorthair-mackerel-tabby_01-idle-blink_seed-20000.mp4
```

在命令末尾加 `--dry-run` 可先核对实际路径、seed 和输出名。
详细筛选命令与编号对照见 [V5 运行说明](../multiseed_v5/README.md)。
