python3 -m sglang.launch_server \
--model /workspace/data/Meta-Llama-3-8B-Instruct/ \
--tp 2 \
--disable-overlap-schedule \
--port 30013 \
--mem-fraction-static 0.8 \
--chunked-prefill-size -1 \
--enable-pdmux \
--log-level debug
