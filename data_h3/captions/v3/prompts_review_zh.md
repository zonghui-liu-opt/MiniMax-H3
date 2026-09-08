# 猫咪动作 v3 提示词审阅稿

本轮重写 8 个动作，4 个拨球／舔爪动作保留 v2 原文作回归对照。英文 prompt 遵循 H3 官方基础模式结构。输入模式、时长与尾帧条件列在各条前；`catloop_` 仅为沿用的动作 ID，不代表每条仍要闭环。尚未完成内网生成验证。

吃饭、打滚、入睡暂按可不复位设计，等待可选偏好确认。这里的参考视频只用于人工分析动作，没有传入生成请求。

## 01 · 自然待机：清晰慢眨眼＋轻微呼吸

模式：FL2VA；请求时长：4.46 秒；条件帧：[0, -1]；状态：rewritten_unvalidated。

主动作是完整闭眼再睁开；伴随轻微头部姿态变化和肩胸呼吸，最后自然回到首图。

```text
How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns with the 0.00-second mark of the target video; Picture 2 (from Shot 1) aligns with the 4.46-second mark of the target video.

integrated_multimodal_description: [Shot 1] Live-action studio pet footage. A medium full-body portrait begins with the same seated cat, framing, floor, background and lighting shown in the first frame. Preserve the individual cat's facial proportions, eye colors, coat or bare-skin appearance, markings, ear shape and existing tail anatomy as its pose changes. The camera holds a static shot throughout, and contact shadows follow the cat's weight naturally. Only the cat occupies the scene; the surrounding floor stays clear throughout. The main action is one clearly visible, relaxed slow blink. The cat watches the camera with a soft, attentive expression while its chest and shoulders move gently with breathing. During the first second, the chin tips slightly down and the gaze makes a tiny adjustment. Between 1.00 and 1.80 seconds, both eyelids close all the way, remain fully closed for a brief readable beat, then reopen smoothly; both eyes are visibly open again at the end of this blink. As the eyes reopen, the head makes a small, relaxed correction and the muzzle softens. The paws continue to support the seated body while the shoulders settle with the exhale. During the final second, its small head and gaze adjustments ease naturally into the seated pose and expression in Picture 2, reaching that reference at the final instant. Gentle breathing continues during the approach. 

overall_soundscape: Quiet studio room tone with faint breathing from the cat.

non_diegetic_music: N/A
```

## 02 · 左耳尖轻牵引＋自然回弹

模式：FL2VA；请求时长：4.46 秒；条件帧：[0, -1]；状态：rewritten_unvalidated。

猫自身left耳、画面right侧：耳尖向外略向上被牵引，短促释放，头部与眼睑有轻微反应。

```text
How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns with the 0.00-second mark of the target video; Picture 2 (from Shot 1) aligns with the 4.46-second mark of the target video.

integrated_multimodal_description: [Shot 1] Live-action studio pet footage. A medium full-body portrait begins with the same seated cat, framing, floor, background and lighting shown in the first frame. Preserve the individual cat's facial proportions, eye colors, coat or bare-skin appearance, markings, ear shape and existing tail anatomy as its pose changes. The camera holds a static shot throughout, and contact shadows follow the cat's weight naturally. Only the cat occupies the scene; the surrounding floor stays clear throughout. The main action is one brief, gentle outward tug of the cat's anatomical left ear, on the viewer's right. From about 0.70 to 1.30 seconds, the tip is drawn a short distance diagonally outward and slightly upward, away from the crown of the head toward the viewer's right. The tip leads the motion, the upper ear bends softly behind it, and the ear base stays attached. The ear retains its characteristic shape and normal length; the visible movement is a small outward deflection with the top of the head left clear. At the peak, the head follows by a tiny amount and the eyelids briefly narrow in a mild, relaxed reaction. The tug releases promptly. The ear returns quickly toward its resting angle, makes one much smaller settling movement, and is relaxed again by about 2.30 seconds. The head follows back a little later, the eyes reopen comfortably, and breathing gently moves the chest. The opposite ear makes only a small accompanying orientation adjustment. The cat remains seated and balanced. The interaction is shown entirely through the ear and the cat's reaction; no hand, person, string or tool appears. During the final second, its small head and gaze adjustments ease naturally into the seated pose and expression in Picture 2, reaching that reference at the final instant. Gentle breathing continues during the approach. 

overall_soundscape: Quiet studio room tone and faint cat breathing, with a soft incidental body movement as the head settles.

non_diegetic_music: N/A
```

## 03 · 右耳尖轻牵引＋自然回弹

模式：FL2VA；请求时长：4.46 秒；条件帧：[0, -1]；状态：rewritten_unvalidated。

猫自身right耳、画面left侧：耳尖向外略向上被牵引，短促释放，头部与眼睑有轻微反应。

```text
How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns with the 0.00-second mark of the target video; Picture 2 (from Shot 1) aligns with the 4.46-second mark of the target video.

integrated_multimodal_description: [Shot 1] Live-action studio pet footage. A medium full-body portrait begins with the same seated cat, framing, floor, background and lighting shown in the first frame. Preserve the individual cat's facial proportions, eye colors, coat or bare-skin appearance, markings, ear shape and existing tail anatomy as its pose changes. The camera holds a static shot throughout, and contact shadows follow the cat's weight naturally. Only the cat occupies the scene; the surrounding floor stays clear throughout. The main action is one brief, gentle outward tug of the cat's anatomical right ear, on the viewer's left. From about 0.70 to 1.30 seconds, the tip is drawn a short distance diagonally outward and slightly upward, away from the crown of the head toward the viewer's left. The tip leads the motion, the upper ear bends softly behind it, and the ear base stays attached. The ear retains its characteristic shape and normal length; the visible movement is a small outward deflection with the top of the head left clear. At the peak, the head follows by a tiny amount and the eyelids briefly narrow in a mild, relaxed reaction. The tug releases promptly. The ear returns quickly toward its resting angle, makes one much smaller settling movement, and is relaxed again by about 2.30 seconds. The head follows back a little later, the eyes reopen comfortably, and breathing gently moves the chest. The opposite ear makes only a small accompanying orientation adjustment. The cat remains seated and balanced. The interaction is shown entirely through the ear and the cat's reaction; no hand, person, string or tool appears. During the final second, its small head and gaze adjustments ease naturally into the seated pose and expression in Picture 2, reaching that reference at the final instant. Gentle breathing continues during the approach. 

overall_soundscape: Quiet studio room tone and faint cat breathing, with a soft incidental body movement as the head settles.

non_diegetic_music: N/A
```

## 04 · 右前爪拨毛线球（沿用 v2）

模式：FL2VA；请求时长：4.46 秒；条件帧：[0, -1]；状态：unchanged_v2_regression。

本轮未报告问题，prompt 文本原样保留为回归对照。

```text
How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns with the 0.00-second mark of the target video; Picture 2 (from Shot 1) aligns with the 4.46-second mark of the target video.

integrated_multimodal_description: [Shot 1] Live-action, photorealistic studio pet footage. Picture 1 and Picture 2 show the identical full-frame endpoint image. A locked camera matching the reference perspective frames a medium full-body, front-facing portrait. At 0.00 seconds, the cat exactly matches Picture 1. Throughout the shot, preserve the same cat's facial proportions; each anatomical eye's reference color; exact ear morphology; coat or skin-surface texture; intrinsic marking topology; body proportions; paw appearance; and reference tail state, including a long tail, short tail, visible stump, or no visible tail. These features may change screen projection only as required by the pose and return to the exact endpoint projection. The camera holds a perfectly Static Shot; perspective, focal length, subject scale, crop, focus, exposure, white balance, lighting, background, and floor appearance established by the pictures remain fixed. Physical shadows stay grounded, respond coherently to contact and weight transfer, and return to the endpoint shadow. The cat remains fully inside the frame with clear margins. The clean frame contains only the cat and one transient wool ball defined in the motion path; only this prop unit and its associated contact shadow may cross a frame boundary along the specified entry or exit path. During 0.00–0.40 seconds, the cat holds the endpoint pose. At 0.40 seconds, one small, tightly wound matte wool ball enters from the viewer's left and rolls left-to-right along a straight path on the foreground floor, with visible rotation and one attached contact shadow. From 0.65 to 1.25 seconds, the cat's eyes track first and the head follows with one small turn; the body transfers a little weight onto the anatomical left front paw on the viewer's right while the anatomical right front paw on the viewer's left lifts. From 1.25 to 1.85 seconds, that lifted paw reaches in one compact arc and its pads meet the top of the ball, causing the ball to decelerate to a brief stop while retaining its size, round boundary, and visibility. From 1.95 to 2.30 seconds, the same paw gives one short controlled nudge toward the viewer's right while beginning to retract. The ball resumes rolling at lower speed and, together with its contact shadow, exits completely beyond the viewer's right edge by 3.20 seconds. The active paw lands at its exact reference point by 3.35 seconds, and the head and gaze recenter by 3.50 seconds. The ball remains one distinct object through normal rolling contact with the floor and the brief paw contact, preserving a clear boundary from the paw and reference tail state after each contact. From 3.60 seconds through the end, the cat holds the endpoint pose while only minute coat-or-skin-surface and shadow residuals, if present, damp to stillness. The wool ball and its associated contact shadow are already fully outside the frame. At the final instant, the cat exactly matches Picture 2 in pose, expression, gaze, anatomy, identity, screen-side feature layout, paw and tail state, shadow, and full-frame composition. One coherent labeled action sequence occurs, with no unrelated behavior.

overall_soundscape: A soft wool ball rolls across the floor from left to right, slows under one muted paw contact, then resumes with a quieter roll before leaving the frame. One light paw-pad landing follows over steady studio room tone.

non_diegetic_music: N/A
```

## 05 · 左前爪拨毛线球（沿用 v2）

模式：FL2VA；请求时长：4.46 秒；条件帧：[0, -1]；状态：unchanged_v2_regression。

本轮未报告问题，prompt 文本原样保留为回归对照。

```text
How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns with the 0.00-second mark of the target video; Picture 2 (from Shot 1) aligns with the 4.46-second mark of the target video.

integrated_multimodal_description: [Shot 1] Live-action, photorealistic studio pet footage. Picture 1 and Picture 2 show the identical full-frame endpoint image. A locked camera matching the reference perspective frames a medium full-body, front-facing portrait. At 0.00 seconds, the cat exactly matches Picture 1. Throughout the shot, preserve the same cat's facial proportions; each anatomical eye's reference color; exact ear morphology; coat or skin-surface texture; intrinsic marking topology; body proportions; paw appearance; and reference tail state, including a long tail, short tail, visible stump, or no visible tail. These features may change screen projection only as required by the pose and return to the exact endpoint projection. The camera holds a perfectly Static Shot; perspective, focal length, subject scale, crop, focus, exposure, white balance, lighting, background, and floor appearance established by the pictures remain fixed. Physical shadows stay grounded, respond coherently to contact and weight transfer, and return to the endpoint shadow. The cat remains fully inside the frame with clear margins. The clean frame contains only the cat and one transient wool ball defined in the motion path; only this prop unit and its associated contact shadow may cross a frame boundary along the specified entry or exit path. During 0.00–0.40 seconds, the cat holds the endpoint pose. At 0.40 seconds, one small, tightly wound matte wool ball enters from the viewer's right and rolls right-to-left along a straight path on the foreground floor, with visible rotation and one attached contact shadow. From 0.65 to 1.25 seconds, the cat's eyes track first and the head follows with one small turn; the body transfers a little weight onto the anatomical right front paw on the viewer's left while the anatomical left front paw on the viewer's right lifts. From 1.25 to 1.85 seconds, that lifted paw reaches in one compact arc and its pads meet the top of the ball, causing the ball to decelerate to a brief stop while retaining its size, round boundary, and visibility. From 1.95 to 2.30 seconds, the same paw gives one short controlled nudge toward the viewer's left while beginning to retract. The ball resumes rolling at lower speed and, together with its contact shadow, exits completely beyond the viewer's left edge by 3.20 seconds. The active paw lands at its exact reference point by 3.35 seconds, and the head and gaze recenter by 3.50 seconds. The ball remains one distinct object through normal rolling contact with the floor and the brief paw contact, preserving a clear boundary from the paw and reference tail state after each contact. From 3.60 seconds through the end, the cat holds the endpoint pose while only minute coat-or-skin-surface and shadow residuals, if present, damp to stillness. The wool ball and its associated contact shadow are already fully outside the frame. At the final instant, the cat exactly matches Picture 2 in pose, expression, gaze, anatomy, identity, screen-side feature layout, paw and tail state, shadow, and full-frame composition. One coherent labeled action sequence occurs, with no unrelated behavior.

overall_soundscape: A soft wool ball rolls across the floor from right to left, slows under one muted paw contact, then resumes with a quieter roll before leaving the frame. One light paw-pad landing follows over steady studio room tone.

non_diegetic_music: N/A
```

## 06 · 向左打滚露肚皮＋空中挥爪

模式：I2VA；请求时长：6.00 秒；条件帧：[0]；状态：rewritten_unvalidated。

向猫自身left侧／画面right侧翻，背部着地、腹部朝上；双前爪离地，其中一爪轻挥一次，结尾保持放松露腹。

```text
For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.

integrated_multimodal_description: [Shot 1] Live-action studio pet footage. A medium full-body portrait begins with the same seated cat, framing, floor, background and lighting shown in the first frame. Preserve the individual cat's facial proportions, eye colors, coat or bare-skin appearance, markings, ear shape and existing tail anatomy as its pose changes. The camera holds a static shot throughout, and contact shadows follow the cat's weight naturally. Only the cat occupies the scene; the surrounding floor stays clear throughout. Over this six-second shot, the cat voluntarily rolls onto its back and exposes its belly in one relaxed, playful movement. During the first two seconds, it softens its elbows, tucks its chin slightly and shifts its hips toward its anatomical left, the viewer's right. The left flank makes gentle floor contact, then the shoulders and pelvis continue rolling together until the back rests on the floor. By about 2.80 seconds, the belly faces upward and remains clearly visible to the camera, both front paws are lifted above the chest with softly bent wrists, and the hind legs rest loosely apart. The cat stays within the frame as its body settles on a compact diagonal. Between about 3.00 and 4.50 seconds, one raised forepaw makes a single loose, short sweep through the air above the belly, then curls back toward the chest while the other remains comfortably bent. Its eyes soften into a brief partial squint and open again, and its head nestles against the floor. A small shoulder-and-hip rock accompanies the paw movement. Through 6.00 seconds, it rests comfortably on its back at a slight angle, belly still exposed and chest breathing gently. Any visible tail follows the pelvis and rests naturally on the floor. The movement is self-initiated, with no person or hand entering the image.

overall_soundscape: Quiet room tone with soft paw shifts, one gentle body contact with the floor and faint breathing during the relaxed belly-up pause.

non_diegetic_music: N/A
```

## 07 · 向右打滚露肚皮＋空中挥爪

模式：I2VA；请求时长：6.00 秒；条件帧：[0]；状态：rewritten_unvalidated。

向猫自身right侧／画面left侧翻，背部着地、腹部朝上；双前爪离地，其中一爪轻挥一次，结尾保持放松露腹。

```text
For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.

integrated_multimodal_description: [Shot 1] Live-action studio pet footage. A medium full-body portrait begins with the same seated cat, framing, floor, background and lighting shown in the first frame. Preserve the individual cat's facial proportions, eye colors, coat or bare-skin appearance, markings, ear shape and existing tail anatomy as its pose changes. The camera holds a static shot throughout, and contact shadows follow the cat's weight naturally. Only the cat occupies the scene; the surrounding floor stays clear throughout. Over this six-second shot, the cat voluntarily rolls onto its back and exposes its belly in one relaxed, playful movement. During the first two seconds, it softens its elbows, tucks its chin slightly and shifts its hips toward its anatomical right, the viewer's left. The right flank makes gentle floor contact, then the shoulders and pelvis continue rolling together until the back rests on the floor. By about 2.80 seconds, the belly faces upward and remains clearly visible to the camera, both front paws are lifted above the chest with softly bent wrists, and the hind legs rest loosely apart. The cat stays within the frame as its body settles on a compact diagonal. Between about 3.00 and 4.50 seconds, one raised forepaw makes a single loose, short sweep through the air above the belly, then curls back toward the chest while the other remains comfortably bent. Its eyes soften into a brief partial squint and open again, and its head nestles against the floor. A small shoulder-and-hip rock accompanies the paw movement. Through 6.00 seconds, it rests comfortably on its back at a slight angle, belly still exposed and chest breathing gently. Any visible tail follows the pelvis and rests naturally on the floor. The movement is self-initiated, with no person or hand entering the image.

overall_soundscape: Quiet room tone with soft paw shifts, one gentle body contact with the floor and faint breathing during the relaxed belly-up pause.

non_diegetic_music: N/A
```

## 08 · 舔左前爪（沿用 v2）

模式：FL2VA；请求时长：4.46 秒；条件帧：[0, -1]；状态：unchanged_v2_regression。

本轮未报告问题，prompt 文本原样保留为回归对照。

```text
How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns with the 0.00-second mark of the target video; Picture 2 (from Shot 1) aligns with the 4.46-second mark of the target video.

integrated_multimodal_description: [Shot 1] Live-action, photorealistic studio pet footage. Picture 1 and Picture 2 show the identical full-frame endpoint image. A locked camera matching the reference perspective frames a medium full-body, front-facing portrait. At 0.00 seconds, the cat exactly matches Picture 1. Throughout the shot, preserve the same cat's facial proportions; each anatomical eye's reference color; exact ear morphology; coat or skin-surface texture; intrinsic marking topology; body proportions; paw appearance; and reference tail state, including a long tail, short tail, visible stump, or no visible tail. These features may change screen projection only as required by the pose and return to the exact endpoint projection. The camera holds a perfectly Static Shot; perspective, focal length, subject scale, crop, focus, exposure, white balance, lighting, background, and floor appearance established by the pictures remain fixed. Physical shadows stay grounded, respond coherently to contact and weight transfer, and return to the endpoint shadow. The cat remains fully inside the frame with clear margins, and the clean frame contains only the cat. During 0.00–0.40 seconds, the cat holds the endpoint pose. From 0.55 to 1.20 seconds, the cat transfers weight onto its anatomical right front paw on the viewer's left and the balanced hindquarters, then lifts its anatomical left front paw on the viewer's right by flexing the wrist and elbow. The lifted paw follows one shallow arc toward the mouth while the head lowers and turns only enough to meet it; the paw remains distinct from the muzzle and all toes remain coherent. At 1.40 seconds, the tongue extends, makes the first visible contact with the raised paw, and retracts. At 1.80 seconds, the tongue repeats one second visible contact and retracts fully as the mouth closes. From 2.00 seconds, the head rises and the paw retraces the same arc. The paw pads meet their exact original floor point by 3.30 seconds, weight recenters across both front paws, and the gaze faces forward by 3.50 seconds. The hind paws retain their reference placement throughout, and the reference tail state remains unchanged. From 3.60 seconds through the end, the cat holds the endpoint pose while only minute coat-or-skin-surface and shadow residuals, if present, damp to stillness. At the final instant, the cat exactly matches Picture 2 in pose, expression, gaze, anatomy, identity, screen-side feature layout, paw and tail state, shadow, and full-frame composition. One coherent labeled action sequence occurs, with no unrelated behavior.

overall_soundscape: Two small wet licking sounds occur separately in sync with the two tongue contacts. A faint body-surface movement and one soft paw-pad landing sit over quiet studio room tone.

non_diegetic_music: N/A
```

## 09 · 舔右前爪（沿用 v2）

模式：FL2VA；请求时长：4.46 秒；条件帧：[0, -1]；状态：unchanged_v2_regression。

本轮未报告问题，prompt 文本原样保留为回归对照。

```text
How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns with the 0.00-second mark of the target video; Picture 2 (from Shot 1) aligns with the 4.46-second mark of the target video.

integrated_multimodal_description: [Shot 1] Live-action, photorealistic studio pet footage. Picture 1 and Picture 2 show the identical full-frame endpoint image. A locked camera matching the reference perspective frames a medium full-body, front-facing portrait. At 0.00 seconds, the cat exactly matches Picture 1. Throughout the shot, preserve the same cat's facial proportions; each anatomical eye's reference color; exact ear morphology; coat or skin-surface texture; intrinsic marking topology; body proportions; paw appearance; and reference tail state, including a long tail, short tail, visible stump, or no visible tail. These features may change screen projection only as required by the pose and return to the exact endpoint projection. The camera holds a perfectly Static Shot; perspective, focal length, subject scale, crop, focus, exposure, white balance, lighting, background, and floor appearance established by the pictures remain fixed. Physical shadows stay grounded, respond coherently to contact and weight transfer, and return to the endpoint shadow. The cat remains fully inside the frame with clear margins, and the clean frame contains only the cat. During 0.00–0.40 seconds, the cat holds the endpoint pose. From 0.55 to 1.20 seconds, the cat transfers weight onto its anatomical left front paw on the viewer's right and the balanced hindquarters, then lifts its anatomical right front paw on the viewer's left by flexing the wrist and elbow. The lifted paw follows one shallow arc toward the mouth while the head lowers and turns only enough to meet it; the paw remains distinct from the muzzle and all toes remain coherent. At 1.40 seconds, the tongue extends, makes the first visible contact with the raised paw, and retracts. At 1.80 seconds, the tongue repeats one second visible contact and retracts fully as the mouth closes. From 2.00 seconds, the head rises and the paw retraces the same arc. The paw pads meet their exact original floor point by 3.30 seconds, weight recenters across both front paws, and the gaze faces forward by 3.50 seconds. The hind paws retain their reference placement throughout, and the reference tail state remains unchanged. From 3.60 seconds through the end, the cat holds the endpoint pose while only minute coat-or-skin-surface and shadow residuals, if present, damp to stillness. At the final instant, the cat exactly matches Picture 2 in pose, expression, gaze, anatomy, identity, screen-side feature layout, paw and tail state, shadow, and full-frame composition. One coherent labeled action sequence occurs, with no unrelated behavior.

overall_soundscape: Two small wet licking sounds occur separately in sync with the two tongue contacts. A faint body-surface movement and one soft paw-pad landing sit over quiet studio room tone.

non_diegetic_music: N/A
```

## 10 · 从容吃猫粮，食盘留在画面内

模式：I2VA；请求时长：8.00 秒；条件帧：[0]；状态：rewritten_unvalidated。

一盘少量猫粮由底边滑入后停稳；低头嗅闻、吃一小口并咀嚼，再吃一口；尾段仍从容进食。取消鼻推食盘和强制出画。

```text
For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.

integrated_multimodal_description: [Shot 1] Live-action studio pet footage. A medium full-body portrait begins with the same seated cat, framing, floor, background and lighting shown in the first frame. Preserve the individual cat's facial proportions, eye colors, coat or bare-skin appearance, markings, ear shape and existing tail anatomy as its pose changes. The camera holds a static shot throughout, and contact shadows follow the cat's weight naturally. The scene contains the cat and, after its entrance, one shallow plain dish with a small portion of dry cat food. Over this eight-second shot, the cat eats at an unhurried pace. During the first 1.30 seconds, the dish slides gently into view from the bottom edge along the floor and slows to a complete stop within comfortable reach, just ahead of the front paws. The dish and its food remain a single stable prop on the floor. The cat notices it with a small downward gaze shift, the ears orient slightly toward it, and the head follows. Between about 1.30 and 3.00 seconds, the cat lowers its head comfortably and briefly sniffs just above the food. Between 3.00 and 5.50 seconds, it takes one small mouthful, lifts its muzzle slightly clear of the dish and chews in a few small jaw movements with natural pauses. Its eyes soften, and the neck and shoulders adjust subtly while the paws support a comfortable seated posture. Between 5.50 and 8.00 seconds, it lowers its muzzle for another small mouthful and continues eating calmly. The last moment shows the cat still comfortably attending to the food, with the dish resting in the same place. Only the dish and food enter the scene; no person or hand appears.

overall_soundscape: Quiet studio ambience, a soft dish slide at the start, a faint sniff and small, separated crunching sounds during eating.

non_diegetic_music: N/A
```

## 11 · 向左收身蜷卧入睡

模式：I2VA；请求时长：6.00 秒；条件帧：[0]；状态：rewritten_unvalidated。

头与肩向画面right侧小幅转动，前爪留在身下、胸腹下降、后腿收拢成紧凑椭圆；头枕前爪闭眼，结尾保持睡姿。

```text
For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.

integrated_multimodal_description: [Shot 1] Live-action studio pet footage. A medium full-body portrait begins with the same seated cat, framing, floor, background and lighting shown in the first frame. Preserve the individual cat's facial proportions, eye colors, coat or bare-skin appearance, markings, ear shape and existing tail anatomy as its pose changes. The camera holds a static shot throughout, and contact shadows follow the cat's weight naturally. Only the cat occupies the scene; the surrounding floor stays clear throughout. Over this six-second shot, the seated cat curls into a compact sleeping pose. During the first second, the eyelids grow heavy and the head dips softly. From about 1.00 to 3.50 seconds, the head and shoulders turn a little toward the cat's anatomical left, the viewer's right, as both forelegs fold close beneath the chest. A short paw adjustment stays directly under the body. The chest lowers vertically toward those supporting paws while the hindquarters settle beside it and the hind legs tuck inward. The spine rounds into a small C shape, bringing the shoulders and hips into one compact oval footprint. The belly faces the floor and the flank rests gently against it. The front paws stay gathered beneath the face throughout the descent. Any visible tail follows its existing length and curls only as far as that anatomy allows along the outside of the body. Between about 3.50 and 4.50 seconds, the chin rests on the gathered front paws near the viewer's right side of this curled shape, the eyelids close fully and the shoulders relax. From 4.50 to 6.00 seconds, the cat remains curled up with its head down and eyes closed, showing a small, steady rise and fall of the flank. The descent is continuous and comfortably paced, followed by a short sleeping hold.

overall_soundscape: Quiet studio room tone with small paw adjustments and a soft body contact as the cat settles, followed by calm, faint breathing.

non_diegetic_music: N/A
```

## 12 · 向右收身蜷卧入睡

模式：I2VA；请求时长：6.00 秒；条件帧：[0]；状态：rewritten_unvalidated。

头与肩向画面left侧小幅转动，前爪留在身下、胸腹下降、后腿收拢成紧凑椭圆；头枕前爪闭眼，结尾保持睡姿。

```text
For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.

integrated_multimodal_description: [Shot 1] Live-action studio pet footage. A medium full-body portrait begins with the same seated cat, framing, floor, background and lighting shown in the first frame. Preserve the individual cat's facial proportions, eye colors, coat or bare-skin appearance, markings, ear shape and existing tail anatomy as its pose changes. The camera holds a static shot throughout, and contact shadows follow the cat's weight naturally. Only the cat occupies the scene; the surrounding floor stays clear throughout. Over this six-second shot, the seated cat curls into a compact sleeping pose. During the first second, the eyelids grow heavy and the head dips softly. From about 1.00 to 3.50 seconds, the head and shoulders turn a little toward the cat's anatomical right, the viewer's left, as both forelegs fold close beneath the chest. A short paw adjustment stays directly under the body. The chest lowers vertically toward those supporting paws while the hindquarters settle beside it and the hind legs tuck inward. The spine rounds into a small C shape, bringing the shoulders and hips into one compact oval footprint. The belly faces the floor and the flank rests gently against it. The front paws stay gathered beneath the face throughout the descent. Any visible tail follows its existing length and curls only as far as that anatomy allows along the outside of the body. Between about 3.50 and 4.50 seconds, the chin rests on the gathered front paws near the viewer's left side of this curled shape, the eyelids close fully and the shoulders relax. From 4.50 to 6.00 seconds, the cat remains curled up with its head down and eyes closed, showing a small, steady rise and fall of the flank. The descent is continuous and comfortably paced, followed by a short sleeping hold.

overall_soundscape: Quiet studio room tone with small paw adjustments and a soft body contact as the cat settles, followed by calm, faint breathing.

non_diegetic_music: N/A
```
