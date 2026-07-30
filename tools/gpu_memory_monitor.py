import argparse
import csv
import signal
import time

import pynvml


running = True


def stop_monitor(*_):
    global running
    running = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--interval", type=float, default=0.5)
    args = parser.parse_args()

    signal.signal(signal.SIGINT, stop_monitor)
    signal.signal(signal.SIGTERM, stop_monitor)
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(args.gpu)

    with open(args.output, "w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow([
            "timestamp", "gpu", "gpu_used_mib", "gpu_free_mib", "gpu_total_mib",
            "gpu_util_pct", "memory_util_pct", "power_w", "temperature_c",
            "compute_processes",
        ])
        while running:
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            try:
                power_w = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
            except pynvml.NVMLError:
                power_w = None
            try:
                processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                process_text = ";".join(
                    f"{process.pid}:{process.usedGpuMemory / (1024**2):.1f}MiB"
                    for process in processes
                )
            except pynvml.NVMLError:
                process_text = ""
            writer.writerow([
                time.time(), args.gpu, memory.used / (1024**2), memory.free / (1024**2),
                memory.total / (1024**2), utilization.gpu, utilization.memory, power_w,
                pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU),
                process_text,
            ])
            output.flush()
            time.sleep(args.interval)
    pynvml.nvmlShutdown()


if __name__ == "__main__":
    main()
