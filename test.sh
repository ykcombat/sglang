
curl http://localhost:30013/flush_cache
python3 -m sglang.bench_serving --backend sglang --port 30013 --num-prompts 300 --random-input 1024 --request-rate 6 