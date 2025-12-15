# SGLANG_PORT=30015
# 64 128 256 512 1024
# CHUNK_SIZE=1024
python3 -m sglang.launch_server \
--model /workspace/data/CodeLlama-34b-Instruct-hf \
--tp 1 \
--disable-overlap-schedule \
--port $SGLANG_PORT \
--chunked-prefill-size $CHUNK_SIZE \
--enable-mixed-chunk
