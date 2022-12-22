# Requires psutil loguru nvidia-ml-py3
import torch
import psutil
import atexit
import nvidia_smi
import os

DEBUG = True


def init_gpu_logging():
    nvidia_smi.nvmlInit()
    atexit.register(nvidia_smi.nvmlShutdown)


def memory_use():
    mem = psutil.virtual_memory()
    swap = psutil.swap_memory()
    available_memory = mem.available
    used_memory_percent = mem.percent
    available_swap = swap.free
    used_swap_percent = swap.percent
    mes = (
        f"The system has {available_memory} free memory (used {used_memory_percent}%),"
        + f"and {available_swap} swap (used {used_swap_percent}%)"
    )
    return mes


def disk_use(path):
    disk_usage = psutil.disk_usage(path)
    free_space = disk_usage.free
    percent_used = disk_usage.percent
    mes = f"The disk that contains {path} has {free_space} free_space and is {percent_used}% full\n"
    return mes


def cpu_use_detailed_nonblocking():
    """Update and/or return the per CPU list using the psutil library."""
    # The first value is bogus, as it meassures the cpu activity since the last call.
    percpu_percent = []
    for cpu_number, cputimes in enumerate(
        psutil.cpu_times_percent(interval=0.0, percpu=True)
    ):
        cpu = {
            "key": "f{cpu_number}",
            "cpu_number": cpu_number,
            "total": round(100 - cputimes.idle, 1),
            "user": cputimes.user,
            "system": cputimes.system,
            "idle": cputimes.idle,
        }
        if hasattr(cputimes, "nice"):
            cpu["nice"] = cputimes.nice
        if hasattr(cputimes, "iowait"):
            cpu["iowait"] = cputimes.iowait
        if hasattr(cputimes, "irq"):
            cpu["irq"] = cputimes.irq
        if hasattr(cputimes, "softirq"):
            cpu["softirq"] = cputimes.softirq
        if hasattr(cputimes, "steal"):
            cpu["steal"] = cputimes.steal
        if hasattr(cputimes, "guest"):
            cpu["guest"] = cputimes.guest
        if hasattr(cputimes, "guest_nice"):
            cpu["guest_nice"] = cputimes.guest_nice
        percpu_percent.append(cpu)
    return percpu_percent


def cpu_use_detailed_blocking_for_1sec():
    """Update and/or return the per CPU list using the psutil library."""
    # Carefull this may sleep for one second. I think this has too much overhead.
    percpu_percent = []
    for cpu_number, cputimes in enumerate(
        psutil.cpu_times_percent(interval=1.0, percpu=True)
    ):
        cpu = {
            "key": "f{cpu_number}",
            "cpu_number": cpu_number,
            "total": round(100 - cputimes.idle, 1),
            "user": cputimes.user,
            "system": cputimes.system,
            "idle": cputimes.idle,
        }
        if hasattr(cputimes, "nice"):
            cpu["nice"] = cputimes.nice
        if hasattr(cputimes, "iowait"):
            cpu["iowait"] = cputimes.iowait
        if hasattr(cputimes, "irq"):
            cpu["irq"] = cputimes.irq
        if hasattr(cputimes, "softirq"):
            cpu["softirq"] = cputimes.softirq
        if hasattr(cputimes, "steal"):
            cpu["steal"] = cputimes.steal
        if hasattr(cputimes, "guest"):
            cpu["guest"] = cputimes.guest
        if hasattr(cputimes, "guest_nice"):
            cpu["guest_nice"] = cputimes.guest_nice
        percpu_percent.append(cpu)
    return percpu_percent


def cpu_use_blocking_1sec():
    # Carefull this may sleep for one second. I think this has too much overhead.
    lst = psutil.cpu_percent(interval=1.0, percpu=True)
    usage_list = []
    for idx, elem in enumerate(lst):
        usage = f"Cpu {idx} has a usage of {elem}\n"
        usage_list.append(usage)
    res = "".join(usage_list)
    return res


def cpu_use_nonblocking():
    # The first value is bogus, as it meassures the cpu activity since the last call.
    lst = psutil.cpu_percent(interval=0.0, percpu=True)
    usage_list = []
    for idx, elem in enumerate(lst):
        usage = f"Cpu {idx} has a usage of {elem}\n"
        usage_list.append(usage)
    res = "".join(usage_list)
    return res


def cpu_use():
    return cpu_use_nonblocking()


def gpu_use():
    if torch.cuda.is_available():
        deviceCount = nvidia_smi.nvmlDeviceGetCount()
        gpu_usage = []
        for i in range(deviceCount):
            handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
            info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
            usage = "Device {}: {}, Memory : ({:.2f}% free): {} (total), {} (free), {} (used) \n".format(
                i,
                nvidia_smi.nvmlDeviceGetName(handle),
                100 * info.free / info.total,
                info.total,
                info.free,
                info.used,
            )
            gpu_usage.append(usage)

        if torch.cuda.device_count() == 1:
            used_device = torch.cuda.current_device()
            current_handle = nvidia_smi.nvmlDeviceGetHandleByIndex(used_device)
            current_info = nvidia_smi.nvmlDeviceGetMemoryInfo(current_handle)
            current_usage = (
                "The GPU used is {} {}: {}, Memory : ({:.2f}% free)\n".format(
                    used_device,
                    nvidia_smi.nvmlDeviceGetName(current_handle),
                    100 * current_info.free / current_info.total,
                )
            )
            gpu_usage.append(current_usage)
        else:
            pass

        return "".join(gpu_usage)
    else:
        return "No NVIDIA GPU avalible, using CPU."


def do_logging_system_use(
    logger,
    current_epoch,
    current_dir,
    data_dir,
):
    assert logger is not None
    logger.debug(f"Finished epoch {current_epoch}")
    logger.debug(cpu_use())
    logger.debug(memory_use())
    logger.debug(gpu_use())
    logger.debug("Current disk:" + disk_use(current_dir))
    logger.debug("Data disk:" + disk_use(data_dir))
