"""
Mixin class providing multiplexing scheduling logic
"""

import logging
import time

import torch
import torch.distributed as dist
from torch.cuda.streams import ExternalStream

from sglang.srt.utils import require_mlp_sync
from sglang.srt.distributed.parallel_state import set_pdmux_status, get_tp_group
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.multiplex.pdmux_context import (
    get_current_stream_idx,
    set_current_stream_idx,
)

logger = logging.getLogger(__name__)


class SchedulerMultiplexDPAttnMixin:

    def adjust_stream_groups(self) -> tuple[int, tuple[ExternalStream, ExternalStream]]:
        decode_bs = sum(self.running_batch.global_num_tokens) if self.running_batch.global_num_tokens else 0

        prefill_tokens = sum(self.split_prefill_batch.global_num_tokens) if self.split_prefill_batch and self.split_prefill_batch.global_num_tokens else 0

        if decode_bs > 0 and prefill_tokens > 0:
            manual_divisions = self.pdmux_config.manual_divisions
            if manual_divisions:
                for i in range(len(manual_divisions)):
                    _, _, threshold = manual_divisions[i]
                    if decode_bs >= threshold:
                        stream_idx = i + 1
            else:
                stream_idx = max(
                    1,
                    min(
                        self.real_sm_group_num - 2,
                        decode_bs
                        * (self.real_sm_group_num - 2)
                        // self.pdmux_config.decode_bs_divisor,
                    ),
                )
            set_current_stream_idx(stream_idx)
        elif decode_bs > 0:
            set_current_stream_idx(self.real_sm_group_num - 1)
        else:
            set_current_stream_idx(0)

        stream_idx = get_current_stream_idx()

        self.tp_worker.model_runner.update_decode_attn_backend(stream_idx)
        return stream_idx, self.stream_groups[stream_idx]
    
    def update_split_prefill_batch_dp(self, sm_count: int) -> bool:
        if self.split_prefill_batch:
            return False

        # add new request
        batch = self.get_new_batch_prefill()
        # sync dp rank
        batch = self.prepare_mlp_sync_batch(batch, tp_group=get_tp_group())
        
        if batch:
            if not batch.is_empty():
                logger.debug(f"NON-EMPTY PREFILL {batch.global_num_tokens =}")
                batch.forward_mode = ForwardMode.SPLIT_PREFILL
            else:
                logger.debug(f"IDLE PREFILL {batch.global_num_tokens =}")
                batch.forward_mode = ForwardMode.SPLIT_PREFILL_IDLE
            self.split_prefill_batch = batch
            return True
        return False

    @torch.inference_mode()
    def event_loop_pdmux_dp_attn(self):
        """A scheduler loop for pd multiplexing."""
        decode_done = False
        prefill_done = False
        wait_prefill_kernel_done = False
        adjust_stream_group = False
        # set_current_stream_idx(1) # debug set 1
        stream_idx = get_current_stream_idx()
        stream_group = self.stream_groups[stream_idx]
        prefill_stream = stream_group[0]
        decode_stream = stream_group[1]
        torch.cuda.empty_cache()

        logger.debug("Starting event loop for pd multiplexing with dp attention ...")
        logger.debug(
            f"PDMux stream groups: {stream_idx}, prefill sm: {self.sm_counts[stream_idx][0]}, decode sm: {self.sm_counts[stream_idx][1]}"
        )

        while True:
            with torch.cuda.stream(decode_stream):
                set_pdmux_status(False)
                recv_reqs = self.recv_requests()
                self.process_input_requests(recv_reqs)

            with torch.cuda.stream(prefill_stream):
                set_pdmux_status(True)
                sm_count = self.sm_counts[stream_idx][0]
                if not wait_prefill_kernel_done:
                    adjust_stream_group = (
                        self.update_split_prefill_batch_dp(sm_count) or adjust_stream_group
                    )

            with torch.cuda.stream(decode_stream):
                set_pdmux_status(False)
                self.running_batch = self.update_running_batch(self.running_batch)
                # Handle DP attn preparetion
                batch = self.running_batch if not self.running_batch.is_empty() else None
                batch = self.prepare_mlp_sync_batch(batch)
                if batch:
                    if not batch.is_empty():
                        pass
                        # logger.debug(f"NON-EMPTY DECODE {batch.global_num_tokens =}")
                    else:
                        pass
                        # logger.debug(f"IDLE DECODE {batch.global_num_tokens =}")
                
                self.running_batch = batch or self.running_batch


                adjust_stream_group = adjust_stream_group or (
                    stream_idx > 0 and self.running_batch.is_empty()
                )
                if not batch and self.split_prefill_batch is None:
                    self.check_memory()
                    self.check_tree_cache()
                    self.new_token_ratio = self.init_new_token_ratio
                    self.maybe_sleep_on_idle()

            if adjust_stream_group:
                prefill_stream.synchronize()
                decode_stream.synchronize()
                stream_idx, stream_group = self.adjust_stream_groups()
                prefill_stream = stream_group[0]
                decode_stream = stream_group[1]
                adjust_stream_group = False
                logger.debug(
                    f"Adjusting stream groups: {stream_idx}, prefill sm: {self.sm_counts[stream_idx][0]}, decode sm: {self.sm_counts[stream_idx][1]}"
                )

            with torch.cuda.stream(decode_stream):
                set_pdmux_status(False)
                # process decode batch
                if not self.running_batch.is_empty():
                    # logger.debug("run decode batch")
                    decode_result = self.run_batch(self.running_batch)
                    decode_done = True
                elif self.running_batch.forward_mode and self.running_batch.forward_mode.is_idle():
                    # logger.debug("run decode IDLE batch")
                    decode_result = self.run_batch(self.running_batch)
                    decode_done = True
                else:
                    decode_done = False
            with torch.cuda.stream(prefill_stream):
                set_pdmux_status(True)
                if (
                    self.split_prefill_batch
                    and (not self.split_prefill_batch.is_empty() or self.split_prefill_batch.forward_mode == ForwardMode.SPLIT_PREFILL_IDLE)
                    and not wait_prefill_kernel_done
                ):
                    prefill_done = True
                    forward_count = (
                        max(
                            1,
                            self.pdmux_config.split_forward_token_budget
                            // sum(self.split_prefill_batch.global_num_tokens),
                        )
                        if self.split_prefill_batch.extend_num_tokens > 0
                        else self.model_config.num_hidden_layers
                    )
                    next_split_index = min(
                        self.split_prefill_batch.split_index + forward_count,
                        self.model_config.num_hidden_layers,
                    )
                    forward_count = (
                        next_split_index - self.split_prefill_batch.split_index
                    )

                    self.split_prefill_batch.split_forward_count = forward_count
                    logger.debug(
                        f"run split prefill batch from {self.split_prefill_batch.split_index} to {next_split_index}, forward count: {forward_count}"
                    )
                    prefill_result = self.run_batch(self.split_prefill_batch)
                    if next_split_index == self.model_config.num_hidden_layers:
                        self.split_prefill_batch.split_prefill_finished = True
                        prefill_exe_done = prefill_stream.record_event()
                    self.split_prefill_batch.split_index = next_split_index

                elif wait_prefill_kernel_done:
                    prefill_done = True
                else:
                    prefill_done = False

            with torch.cuda.stream(decode_stream):
                set_pdmux_status(False)
                decode_stream.synchronize()
                if decode_done:
                    self.process_batch_result(self.running_batch, decode_result)
                    self.running_batch.forward_mode = ForwardMode.DECODE # reset to decode mode

            with torch.cuda.stream(prefill_stream):
                set_pdmux_status(True)
                if prefill_done and self.split_prefill_batch.split_prefill_finished:
                    wait_prefill_kernel_done = True
                    prefill_exe_done_flag = prefill_exe_done.query()
                    flags = (
                        torch.ones(1, device="cpu", dtype=torch.int32)
                        if prefill_exe_done_flag
                        else torch.zeros(1, device="cpu", dtype=torch.int32)
                    )

                    self.tp_cpu_group.allreduce(flags, dist.ReduceOp.SUM).wait()
                    if flags.item() == self.tp_size:
                        self.process_batch_result(
                            self.split_prefill_batch, prefill_result
                        )

                        if not self.split_prefill_batch.forward_mode.is_idle():
                            if self.running_batch and not self.running_batch.is_empty():
                                self.running_batch.merge_batch(self.split_prefill_batch)
                            else:
                                self.running_batch = self.split_prefill_batch

                        self.split_prefill_batch = None
                        wait_prefill_kernel_done = False
                        adjust_stream_group = True
