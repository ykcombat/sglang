SGLANG_PORT=30015


# sharegpt
OUTPUT_FILE="sharegpt_muxwise.jsonl"
for ((i=1;i<=25;i++)); do
    curl http://localhost:${SGLANG_PORT}/flush_cache
    python3 -m sglang.bench_serving --backend sglang --port $SGLANG_PORT --num-prompts 500 --request-rate $i --output-file $OUTPUT_FILE
done


# loogle
OUTPUT_FILE="loogle_muxwise.jsonl"
for ((i=1;i<=15;i++)); do
    rate=$(echo "$i * 0.04 + 0.04" | bc)
    curl http://localhost:$SGLANG_PORT/flush_cache
    python benchmark/pdmux/bench_serving.py --dataset-name loogle --num-prompts 20 --model /workspace/data/CodeLlama-34b-Instruct-hf --backend sglang --request-rate $rate --port $SGLANG_PORT --output-file $OUTPUT_FILE
done
