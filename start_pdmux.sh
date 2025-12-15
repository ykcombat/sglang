# SGLANG_PORT=30015

python3 -m sglang.launch_server \
--model /workspace/data/CodeLlama-34b-Instruct-hf \
--tp 1 \
--disable-overlap-schedule \
--port $SGLANG_PORT \
--mem-fraction-static 0.8 \
--chunked-prefill-size -1 \
--enable-pdmux \
--pdmux-config-path /workspace/sglang/sharegpt.yml \
--log-level debug
