import atexit
import json
import os
import resource
import time
from pathlib import Path

import torch


_enabled = False
_finished = False
_events_path = None
_snapshot_path = None
_summary_path = None
_last_stage = None
_start_time = None


def _tensor_storages(obj):
    seen_objects = set()
    storages = {}

    def visit(value):
        if value is None:
            return
        object_id = id(value)
        if object_id in seen_objects:
            return
        seen_objects.add(object_id)

        if torch.is_tensor(value):
            try:
                storage = value.untyped_storage()
                key = (value.device.type, value.device.index, storage.data_ptr())
                storages[key] = max(storages.get(key, 0), storage.nbytes())
            except RuntimeError:
                pass
            return
        if isinstance(value, torch.nn.Module):
            for parameter in value.parameters():
                visit(parameter)
                visit(parameter.grad)
            for buffer in value.buffers():
                visit(buffer)
            return
        if isinstance(value, torch.optim.Optimizer):
            for group in value.param_groups:
                visit(group.get("params"))
            visit(value.state)
            return
        if isinstance(value, dict):
            for item in value.values():
                visit(item)
            return
        if isinstance(value, (list, tuple, set)):
            for item in value:
                visit(item)

    visit(obj)
    totals = {"cpu": 0, "cuda": 0}
    for (device_type, _, _), nbytes in storages.items():
        if device_type in totals:
            totals[device_type] += nbytes
    return {f"{key}_mib": value / (1024**2) for key, value in totals.items()}


def _object_memory(obj):
    result = _tensor_storages(obj)
    result["type"] = type(obj).__name__
    if isinstance(obj, torch.nn.Module):
        result["parameters"] = _tensor_storages(list(obj.parameters()))
        result["gradients"] = _tensor_storages(
            [parameter.grad for parameter in obj.parameters()]
        )
        result["buffers"] = _tensor_storages(list(obj.buffers()))
    elif isinstance(obj, torch.optim.Optimizer):
        result["parameter_groups"] = _tensor_storages(
            [group.get("params", []) for group in obj.param_groups]
        )
        result["optimizer_state"] = _tensor_storages(obj.state)
    return result


def _current_rss_mib():
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        with open("/proc/self/statm", "r", encoding="utf-8") as handle:
            resident_pages = int(handle.read().split()[1])
        return resident_pages * page_size / (1024**2)
    except (OSError, ValueError, IndexError):
        # ru_maxrss is KiB on Linux and bytes on macOS. The server path above
        # is preferred; this is only a portable fallback.
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return value / 1024.0


def _write_event(event):
    if _events_path is None:
        return
    with _events_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")


def _collect_event(stage, objects, synchronize):
    if synchronize:
        torch.cuda.synchronize()
    stats = torch.cuda.memory_stats()
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    driver_used = total_bytes - free_bytes
    inactive_split = stats.get("inactive_split_bytes.all.current", 0)
    return {
        "timestamp": time.time(),
        "elapsed_seconds": (
            time.time() - _start_time if _start_time is not None else None
        ),
        "stage": stage,
        "previous_stage": _last_stage,
        "device": torch.cuda.current_device(),
        "device_name": torch.cuda.get_device_name(torch.cuda.current_device()),
        "allocated_mib": allocated / (1024**2),
        "reserved_mib": reserved / (1024**2),
        "allocator_slack_mib": (reserved - allocated) / (1024**2),
        "driver_used_mib": driver_used / (1024**2),
        "non_torch_or_context_mib": max(0, driver_used - reserved) / (1024**2),
        "max_allocated_mib": torch.cuda.max_memory_allocated() / (1024**2),
        "max_reserved_mib": torch.cuda.max_memory_reserved() / (1024**2),
        "active_mib": stats.get("active_bytes.all.current", 0) / (1024**2),
        "inactive_split_mib": inactive_split / (1024**2),
        "inactive_split_ratio": inactive_split / reserved if reserved else 0.0,
        "cuda_free_mib": free_bytes / (1024**2),
        "cuda_total_mib": total_bytes / (1024**2),
        "process_rss_mib": _current_rss_mib(),
        "num_alloc_retries": stats.get("num_alloc_retries", 0),
        "num_ooms": stats.get("num_ooms", 0),
        "objects": {name: _object_memory(value) for name, value in objects.items()},
    }


def configure_memory_profiler(outdir):
    global _enabled, _finished, _events_path, _snapshot_path, _summary_path
    global _last_stage, _start_time
    _enabled = os.environ.get("QVDIT_MEMORY_PROFILE", "0") == "1" and torch.cuda.is_available()
    if not _enabled:
        return

    _finished = False
    _last_stage = None
    _start_time = time.time()

    profile_dir = Path(outdir) / "memory_profile"
    profile_dir.mkdir(parents=True, exist_ok=True)
    _events_path = profile_dir / "torch_memory_events.jsonl"
    _snapshot_path = profile_dir / "torch_memory_snapshot.pickle"
    _summary_path = profile_dir / "torch_memory_summary.txt"
    _events_path.write_text("", encoding="utf-8")
    torch.cuda.reset_peak_memory_stats()

    if os.environ.get("QVDIT_MEMORY_HISTORY", "1") == "1":
        max_entries = int(os.environ.get("QVDIT_MEMORY_MAX_ENTRIES", "300000"))
        torch.cuda.memory._record_memory_history(
            enabled="all",
            context="all",
            stacks="all",
            max_entries=max_entries,
        )
    atexit.register(finish_memory_profiler)
    mark_memory("profiler_started")


def mark_memory(stage, **objects):
    global _last_stage
    if not _enabled:
        return
    event = _collect_event(stage, objects, synchronize=True)
    _write_event(event)
    _last_stage = stage


def record_memory_oom(error):
    """Persist allocator state after OOM without performing new CUDA work."""
    global _finished, _last_stage
    if not _enabled or _finished:
        return
    stage = f"oom_after_{_last_stage or 'unknown'}"
    try:
        event = _collect_event(stage, {}, synchronize=False)
        event["error"] = repr(error)
        _write_event(event)
        _last_stage = stage
    except Exception as profile_error:
        _write_event({
            "stage": "oom_profiler_event_error",
            "error": repr(profile_error),
            "original_error": repr(error),
        })

    try:
        if _summary_path is not None:
            _summary_path.write_text(
                torch.cuda.memory_summary(abbreviated=False), encoding="utf-8"
            )
    except Exception as summary_error:
        _write_event({"stage": "oom_summary_error", "error": repr(summary_error)})

    try:
        if os.environ.get("QVDIT_MEMORY_HISTORY", "1") == "1":
            torch.cuda.memory._dump_snapshot(str(_snapshot_path))
            torch.cuda.memory._record_memory_history(enabled=None)
    except Exception as snapshot_error:
        _write_event({"stage": "oom_snapshot_error", "error": repr(snapshot_error)})
    _finished = True


def finish_memory_profiler():
    global _finished
    if not _enabled or _finished:
        return
    _finished = True
    try:
        mark_memory("profiler_finished")
        if _summary_path is not None:
            _summary_path.write_text(
                torch.cuda.memory_summary(abbreviated=False), encoding="utf-8"
            )
        if os.environ.get("QVDIT_MEMORY_HISTORY", "1") == "1":
            torch.cuda.memory._dump_snapshot(str(_snapshot_path))
            torch.cuda.memory._record_memory_history(enabled=None)
    except Exception as error:
        if _events_path is not None:
            with _events_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"stage": "profiler_error", "error": repr(error)}) + "\n")
