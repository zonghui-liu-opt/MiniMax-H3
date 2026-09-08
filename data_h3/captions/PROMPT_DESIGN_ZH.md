# MiniMax-H3 Prompt 设计

## 官方 skill 基线

使用官方仓库固定提交 `d21241f0a4b3acbb34c97dae47fa417b7065e438` 中的：

- `skills/h3-prompt-writing/SKILL.md`，SHA-256 `a7000443...e56d0`；
- `skills/h3-prompt-writing/references/base-en.txt`，SHA-256 `2cfebc09...e75fc`。

按 skill 先选择 FL2VA，再固定：首行 alignment instruction、一个空行，以及下面的精确顺序：

```text
integrated_multimodal_description: ...

overall_soundscape: ...

non_diegetic_music: N/A
```

当前 runtime contract 为 107 帧、24 fps，首行末时刻按 `107/24=4.4583` 写成 `4.46-second mark`；不是训练派生层的 3.00 秒。若 H100 预检不再解析 107 帧，必须停批并升级动作时点与模板版本。

## 模板策略

固定骨架只替换三个字段：`END_SECONDS`、`MOTION_PATH`、`SOUNDSCAPE`。身份事实由条件图承担，不向 prompt 注入自由生成的长外观描述。所有动作是单镜头、固定机位、单峰值、自然回坐。

12 个动作正文均保存于 `action_bodies_v1.json`，渲染快照位于 `rendered_107/`。MVS 使用其中 7 个，其余 5 个为 192 条正式校准预先准备。

H3 请求明确不发送：

```text
negative_prompt
guidance_scale / true_cfg_scale / audio_guidance_scale
fps / num_frames / seconds
```

`fps/num_frames` 由 H3 adapter 解析；额外的 CFG/negative 键在当前 H3 adapter 中不是“忽略”，而是会破坏 canonical 请求。排除手、箭头、光标、文字、额外主体和镜头运动的要求均写进正向画面合同。

## Prompt lint

`catloop_pipeline.py lint` 会检查：

- alignment 首行和 `4.46`；
- 首行后空行；
- 三字段顺序；
- `non_diegetic_music: N/A` 结尾；
- 无未替换占位符；
- 请求仍有同 URI 的两个 condition，顺序 `[0,-1]`；
- 请求无 negative/CFG/fps/num_frames；
- 所有方向字段与 `action_id` 配对一致。
