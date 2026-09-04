# Auto uses folding for the single-request recipes below and can
# select data parallel encoding for a compatible TP1 request batch.
sglang serve \
  --model-path /srv/workspace/Kirin_AI_DataLake/models/MiniMax-H3 \
  --num-gpus 4 \
  --tp-size 2 \
  --ulysses-degree 2 \
  --performance-mode speed \
  --host 0.0.0.0 \
  --port 30010 \
  --model-variant fl2va