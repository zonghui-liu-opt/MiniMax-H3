# MiniMax-H3 猫咪闭环动作 Prompt Pack V2

## 1. 交付结论

V2 是一套**新增、可独立审计**的 12 动作 MiniMax-H3 FL2VA prompt pack。它不覆盖 V1，也不自动改写现有 7 动作 MVS 的 candidate manifests，避免旧 prompt/hash 与已经保存的运行证据失配。

主要文件：

- `h3_template_v2.txt`：所有动作共享的 H3 FL2VA 骨架；
- `action_bodies_v2.json`：12 个动作的连续运动路径、声音和方向元数据；
- `rendered_107_v2/*.txt`：12 份可直接提交的完整英文 prompt；
- `prompt_pack_v2.jsonl`：prompt、方向、版本、官方来源和 SHA-256 的逐动作清单；
- `../scripts/render_h3_prompt_pack_v2.py`：渲染、语义合同和快照的 fail-closed 校验器。

## 2. 官方事实与本项目设计

### 2.1 MiniMax 官方硬格式

本包依据 MiniMax-H3 官方提交 `d21241f0a4b3acbb34c97dae47fa417b7065e438`：

- `skills/h3-prompt-writing/SKILL.md`，SHA-256 `a7000443588ca3f145e3b3fd8900f14e0325dc460bd811268fac89a9dc8e56d0`；
- `skills/h3-prompt-writing/references/base-en.txt`，SHA-256 `2cfebc096a6e08370f288d468d90b60f7f9bcb938f94bf090816e910e48e75fc`。

官方 FL2VA 规则保持不变：

1. 首行写 Picture 1/Picture 2 与 0.00/结束时刻的对齐；
2. 首行后一个空行；
3. 字段严格按 `integrated_multimodal_description`、`overall_soundscape`、`non_diegetic_music` 排列；
4. 写首帧到尾帧的**连续路径**，不是重复描述两张静态图；
5. FL2VA 通常优先单镜头；无画外配乐时写 `non_diegetic_music: N/A`；
6. `overall_soundscape` 仍写环境声、接触声和非语言动作声，不能因为训练资产最终去音轨就随意写 `N/A`。

本包使用“首行 + 3 个字段 + 3 个空行”的恰好 7 个物理行，便于逐字快照校验；**恰好 7 行是本项目的序列化约定，不是 MiniMax 官方另加的硬规则**。

官方来源：

- <https://github.com/MiniMax-AI/MiniMax-H3/blob/d21241f0a4b3acbb34c97dae47fa417b7065e438/skills/h3-prompt-writing/SKILL.md>
- <https://github.com/MiniMax-AI/MiniMax-H3/blob/d21241f0a4b3acbb34c97dae47fa417b7065e438/skills/h3-prompt-writing/references/base-en.txt>

### 2.2 本项目为训练数据做的选择

以下不是官方强制语法，而是为 Wan2.2-TI2V-5B 猫动作训练数据做的工程选择：

- 单一连续 `Static Shot`，不调用 pan/zoom/tracking 等镜头能力，避免把相机运动混入动作标签；
- 用可观察的承重、关节路径、接触、惯性、遮挡恢复和道具动力学激发 H3 的时序与物理生成能力；
- 同一条件图分别放在 `frame_index=0` 与 `frame_index=-1`，让模型原生收敛到同一整帧端点；
- 用正向、具体的画面结果约束身份与解剖，不新增 `negative_prompt`、CFG 或伪造的物理控制字段；
- 声音只描述与动作同步的真实微声，不加画外配乐。

固定机位不是少用能力：本任务的目标能力是主体运动、接触、物理连续性、身份保持和闭环收敛，镜头变化会成为训练信号污染。

## 3. V2 相对 V1 的关键升级

### 3.1 对任意 `ids/` 表型采用参考图相对合同

V1 的 `both ears are upright` 和 `returns ... upright` 只适合当前两个竖耳 MVS ID，与 `id64_v1` 中的 Scottish Fold、American Curl 等冲突。V2 不再假设统一耳型、尾长或眼色，而是明确保持：

- 左右眼颜色与左右对应；
- 折耳、卷耳或竖耳的真实参考形态；
- 毛长、毛质、颜色及非对称斑纹的屏幕侧；
- 体型与爪部外观；
- 尾巴的参考状态（长尾、短尾、可见尾桩或不可见尾）及端点屏幕投影。

这也保护当前两个压力 ID：浅色长毛猫的高频鬃毛与 viewer-right 蓬松尾，以及黑猫 viewer-left 脸白斑、偏心胸弯月、viewer-right 白趾和 viewer-left 尾巴。

### 3.2 动作正文从“结果句”升级为可观察因果链

| 动作族 | V2 可观察链 |
|---|---|
| `idle` | 微小吸气 → 一次完整眨眼 → 呼气 → 视觉静止 |
| `pull_ear` | 耳根稳定 → 耳软骨小幅弧形外屈 → 单峰值 → 一次微小阻尼回弹 → 恢复参考耳型 |
| `catch_yarn` | 单球带旋转/接触影入场 → 眼睛先追、头后跟 → 明确承重爪和主动爪 → 爪垫接触使球减速 → 同向轻推 → 球与影完全离场 → 爪/头复位 |
| `roll_and_reach` | 重心侧移 → 肩先降低、骨盆跟随 → 同侧前爪伸出 → 紧凑 partial roll 单峰值 → 逆序复位 → 尾巴回到参考侧 |
| `lick_paw` | 重量交给对侧爪 → 目标腕/肘屈曲抬爪 → 舌头两次独立可见接触 → 舌收回/闭嘴 → 爪原路落回 |
| `eat_cat_food` | 浅盘+内容物作为一个道具 assembly 从下缘低惯性入场 → 盘停在前爪前且不遮爪 → 颈肩弧线低头 → 两次独立小口/咬合 → 鼻尖轻推远侧盘沿启动回程 → 盘、剩余可见猫粮与影原路完全退出 |
| `lie_and_sleep` | 肘屈/胸部降低 → 指定侧肩先着地、骨盆跟随 → 紧凑侧卧 → 闭眼短呼吸 → 睁眼 → 前爪依次撑起 → 逆序回坐 |

所有左右动作同时写猫的解剖方向和观者方向。毛线球以屏幕方向命名，同时独立记录主动爪：

- 左→右：主动爪为猫解剖右前爪，即 viewer-left；
- 右→左：主动爪为猫解剖左前爪，即 viewer-right。

### 3.3 控制 prompt 过载

V2 不堆叠 `beautiful`、`cinematic`、高质量等抽象词，也没有负向关键词清单。公共身份/机位合同只写一次，动作正文占据 prompt 的主要新增信息。当前完整 prompt 为 402–580 个英文词；词数只作为遥测，不设伪造的官方上下限。仍需通过内网 A/B 监控“强身份合同压过动作、导致 no-action”的风险，不能只凭静态格式判断 V2 一定优于 V1。

## 4. 时序合同

本包当前是一个**等待内网 adapter preflight 的 provisional 107-frame profile**，不是已经确认的正式时长合同。四种时长口径必须分开记录：

- API 请求时长：`4.00 s`；
- 预期 raw 帧数与帧率：`107 frames @ 24 fps`；
- 按 `frames/fps` 计算的容器时长：`107/24 = 4.458333... s`；
- 最后一帧 PTS：`106/24 = 4.416667... s`；
- 当前 prompt 对齐末时刻：暂按项目既有容器时长口径四舍五入为 `4.46 s`。

机器清单将该状态明确标记为 `requires_internal_adapter_preflight`。因此当前包可以用于单条/小批 A/B，但在内网确认 H3 wrapper 实际采用 `4.00`、容器时长还是末帧 PTS 作为 Picture 2 对齐基准前，**不要正式批量提交**。

- `0.00–0.40 s`：稳定首帧；
- `0.40–3.55 s`：动作起势、峰值和主要恢复；
- `3.60–4.46 s`：残余位移/毛发运动阻尼归零，精确落到 Picture 2。

这里的 4.46 秒只是**待验证的项目解释**，不是 MiniMax 官方对所有 wrapper 的统一常数。官方材料要求 S.SS 匹配有效视频时长，但不能替内网 wrapper 裁决上述口径。若 dry-run 表明 adapter 以请求时长 `4.00 s` 为准，或实际帧数/fps 不同，必须整体升级模板、动作时点、profile 名和快照；不能只替换首行数字。

## 5. 生成与校验

在工程目录执行。只读检查（也是默认模式）：

```bash
python3 scripts/render_h3_prompt_pack_v2.py \
  --project-root . \
  --official-repo /path/to/MiniMax-H3 \
  --check
```

只有明确修订源文件并准备重新冻结快照时才写入：

```bash
python3 scripts/render_h3_prompt_pack_v2.py \
  --project-root . \
  --official-repo /path/to/MiniMax-H3 \
  --write \
  --force

python3 -m pytest -q tests/test_prompt_pack_v2.py
```

`--write` 强制要求 `--official-repo`；skill/guide 任一 SHA 漂移都会失败。`--check` 不写文件，并要求 12 个快照、完整内容和 JSONL manifest 逐字匹配当前源文件。已有快照变化时，没有显式 `--force` 的写入会拒绝覆盖。

提交时从 `prompt_pack_v2.jsonl` 读取对应 `prompt_text`，并继续使用同一张 768×1344 condition URI 两次，顺序固定 `[0,-1]`。`endpoint_anchors_480x832` 是后处理与评测 anchor，不应误当成另一张不同语义条件图。

`prompt_pack_v2.jsonl` 是按动作组织的通用 prompt profile，故显式标记 `condition_binding_scope=unbound_action_prompt_profile`，不冒充已绑定某个 ID。内网 request/run manifest 必须另外记录 ID、两个 condition slot 的 URI/SHA/尺寸、prompt SHA、seed、model/revision 和服务版本，并在提交前断言两个 slot 的 URI 与 SHA 均相同。

## 6. 内网 A/B 测试建议

### 6.1 第零门：adapter 时长预检

先仅提交 1 个 ID × `idle`，保存完整 request/run manifest 与 raw 返回值，核对两个 condition slot 的 URI/SHA，以及服务解析的 prompt 对齐秒数、实际帧数、fps、容器时长和末帧 PTS。只有 4.46 口径得到验证，才保持当前 profile；否则先升级整包，再进入动作 A/B。

### 6.2 第一门：低成本动作可用性

先用当前两只 MVS 压力 ID，V1/V2、12 动作、同一组 seed 各生成一条：

```text
2 IDs × 12 actions × 1 seed × 2 prompt profiles = 48 raw videos
```

这一门只判断：是否发生正确动作、左右是否正确、是否出现解剖/道具灾难、是否自然开始回收。不要在这一批上不断改 prompt 后又把它当无偏评测。

### 6.3 第二门：特殊表型兼容

至少加入一只卷耳、一只折耳、一只短尾/无明显尾和一只长毛低对比 ID。重点检查参考相对合同是否真的优于 V1 的统一竖耳/尾巴假设。

### 6.4 第三门：冻结校准

选定 profile 后，再按正式的代表 ID ×12 动作 ×固定 seeds 做校准。V1/V2 比较应保持：

- 相同 condition 文件与 SHA；
- 相同 seed、H3 revision、推理步数、flow shift、并行方式和服务版本；
- 相同 raw 媒体合同与人工/VLM rubric；
- 先保存不可变 raw107，再做任何 97/73 派生或 endpoint lock。

逐条至少标记：`no_action`、`wrong_action`、`wrong_direction`、`extra_action`、`identity_drift`、`anatomy_failure`、`prop_not_removed`、`camera_motion`、`background_change`、`endpoint_mismatch`、`late_jump`。按动作等权报告通过率，并单列耳、毛线球、翻滚/卧倒这些高风险族，不能让 `idle` 拉高总分。

## 7. 已知风险与调参边界

- `pull_ear` 仍是“画外轻力、画内无操作者”的项目动作合同，物理因果比自主耳动更难；如果高频出现橡皮耳，可在校准集单独比较 self-initiated ear flex，但不要把两个语义混入同一正式 action ID。
- 毛线球、吃食、翻滚、卧倒包含多个必要子事件；如果 4.46 秒内复位失败，应优先缩短峰值停留或动作幅度，不要删除“道具完全离场”和“逆序回坐”。
- 双端点与强身份合同可能造成静止捷径；因此必须跟踪 `no_action`，不能因端点相似就判合格。
- 如果详细版在多个动作上稳定出现 `no_action`，再新建精简的 V2B profile 做同 ID/同 seed A/B；不在本 V2 快照上边测边删文字，也不设伪官方词数上限。
- prompt 只负责促使 H3 自然收敛。后处理复制首尾像素只能在 raw 已通过收敛审核后执行，不能用来掩盖末段跳变。

## 8. 与现有 MVS 的边界

当前 `mvs_config_v1.json` 和既有 manifests 仍是 V1 prompt + 7 动作 MVS。V2 的 12 个 prompt 全部可渲染、可单独提交，但本次不擅自把 MVS 扩为 12 动作，也不重写旧 candidate hash。内网 A/B 选定 V2 后，应另起 candidate plan/schema/profile 版本接入，而不是原地覆盖 V1 证据。
