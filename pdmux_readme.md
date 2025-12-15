<!-- Thank you for your contribution! Please follow these guidelines to enhance your pull request. If anything is unclear, submit your PR and reach out to maintainers for assistance. Join our Slack community at https://slack.sglang.ai to discuss further. -->
## Motivation
Support PD-Multiplexing
## Modifications
- Scheduler:
  - Implemented pdmux scheduling loop in `srt/multiplex/multiplexing.py`.
  - Manage PD-Multiplexing context as global variables in `srt/multiplex/pdmux_context.py`.
  - Added new fields to ScheduleBatch to support split prefill mode.
  - Added forward_batch_split_prefill() in `tp_worker.py` to support split prefill mode.
- Server Args: Extended `server_args.py` with `--enable-pdmux` and `--pdmux-config-path` flags.
- CudaGraphRunner: Record and replay a set of CUDA graphs for each sm partition.
- pynccl: Add an interface to let the pynccl communicator use the current CUDA stream.
- [Docs updated](https://github.com/ykcombat/sglang/blob/e84aa1bdd055df93e603a46fa6ca5e60afd213f5/docs/advanced_features/pd_multiplexing.md) (keep updating)
## Environment
### Use our prebuilt Docker image
```sh
docker pull combathhhhhh/pdmux:sglpr_torch2.6_bench
```
and the source code in the container:
```sh
cd /workspace/sglang
```
### Build your environment from this PR
Primary code:
- [Current Branch](https://github.com/sgl-project/sglang/pull/10692/commits/dde984e4eef060b5df839d4cefb3855f07390e9b)
Versions of other packages:
- torch: 2.6.0+cu126 (Note that you may build sgl-kernel by yourself under this torch version)
- cuda: 12.6
Hardware:
- NVIDIA H200 NVL (140 GB  132 SMs)
- NVIDIA driver: 580.65.06 (must be greater than 570.xx)
### Commands
**start a pdmux server**
```bash
python3 -m sglang.launch_server \
--model meta-llama/CodeLlama-34b-Instruct-hf \
--tp 1 \
--disable-overlap-schedule \
--port 30000 \
--mem-fraction-static 0.8 \
--chunked-prefill-size -1 \
--enable-pdmux \
--pdmux-config-path pdmux_config.yml # see how to configure below
```
**start a chunked prefill server**
```bash
export CHUNK_SIZE=YOU_CHUNK_SIZE
python3 -m sglang.launch_server \
--model meta-llama/CodeLlama-34b-Instruct-hf \
--tp 1 \
--disable-overlap-schedule \
--port 30000 \
--chunked-prefill-size $CHUNK_SIZE \
--enable-mixed-chunk
```
#### Example Config
H200 config YAML
```yaml
# pdmux_config.yaml
# Number of SM groups to divide the GPU into.
# This includes two default groups:
#   - Group 0: all SMs for prefill
#   - Last group: all SMs for decode
# So the number of manual divisions (below) must be (sm_group_num - 2).
sm_group_num: 8
# Optional manual divisions of SMs.
# Each entry contains:
#   - prefill_sm: number of SMs allocated for prefill
#   - decode_sm: number of SMs allocated for decode
#   - decode_bs_threshold: minimum decode batch size at which this group should be selected
#
# If provided, the number of entries must equal (sm_group_num - 2).
manual_divisions:
  - [112, 20, 1]
  - [104, 28, 5]
  - [96, 36, 10]
  - [80, 52, 15]
  - [64, 68, 20]
  - [56, 76, 25]
# Divisor for default stream index calculation.
# Used to adjust sm_group based on decode_bs when no manual_divisions are provided.
# Formula:
#   stream_idx = max(1, min(sm_group_num - 2, decode_bs * (sm_group_num - 2) // decode_bs_divisor))
decode_bs_divisor: 36
# Maximum token budget for split_forward in the prefill stage.
# Determines how many layers are executed per split_forward.
# Formula:
#   forward_count = max(1, forward_token_budget // extend_num_tokens)
split_forward_token_budget: 65536
```
#### Bench
 sharegpt
```bash
for ((i=1;i<=25;i++)); do
    curl http://localhost:30000/flush_cache
    python3 -m sglang.bench_serving --backend sglang --port 30000 --num-prompts 500 --request-rate $i
done
```
loogle
```bash
for ((i=1;i<=15;i++)); do
    rate=$(echo "$i * 0.04 + 0.04" | bc)
    curl http://localhost:30000/flush_cache
    python benchmark/pdmux/bench_serving.py --dataset-name loogle --num-prompts 20 --model meta-llama/CodeLlama-34b-Instruct-hf --backend sglang --request-rate $rate --port 30000
done
```
NOTE: test loogle with pdmux config
```yaml
sm_group_num: 5
manual_divisions:
  - [80, 52, 1]
  - [64, 68, 5]
  - [56, 76, 10]
```
