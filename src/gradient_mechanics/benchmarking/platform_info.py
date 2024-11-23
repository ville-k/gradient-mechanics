import logging
import sys
import platform
import torch


logger = logging.getLogger(__name__)


def log_hardware_info():
    # CPUs
    logger.info("Number of CPUs: %d", torch.get_num_threads())
    logger.info("Number of Inter-socket CPUs: %d", torch.get_num_interop_threads())
    logger.info("CPU capabilities: %s", torch.backends.cpu.get_cpu_capability())

    # GPUs
    logger.info("Number of GPUs: %d", torch.cuda.device_count())
    for device_id in range(torch.cuda.device_count()):
        logger.info(
            " - Device %d: %s", device_id, torch.cuda.get_device_name(device_id)
        )
        logger.info(
            " - Device %d capability: %s",
            device_id,
            torch.cuda.get_device_capability(device_id),
        )
        logger.info(
            " - Device %d memory: %.2f GB",
            device_id,
            torch.cuda.get_device_properties(device_id).total_memory / 1e9,
        )


def log_software_versions():
    logger.info("OS: %s", sys.platform)
    logger.info("Linux platform: %s", platform.platform())
    logger.info("Linux distribution: %s", platform.freedesktop_os_release())
    logger.info("Python version: %s", sys.version_info)
    logger.info("PyTorch version: %s", torch.__version__)
    logger.info(" - git: %s", torch.version.git_version)

    if torch.cuda.is_available():
        logger.info("CUDA is available")
        logger.info("CUDA version: %s", torch.version.cuda)

        # CUDA backend information
        logger.info(" - allow TF32: %s", torch.backends.cuda.matmul.allow_tf32)
        logger.info(
            " - allow FP16 reduced precision reduction: %s",
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
        )
        logger.info(
            " - allow BF16 reduced precision reduction: %s",
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        )
        logger.info(
            " - preferred linalg library: %s",
            torch.backends.cuda.preferred_linalg_library(),
        )

        # cuDNN backend information
        logger.info("cuDNN version: %s", torch.backends.cudnn.version())
        logger.info(" - cuDNN available: %s", torch.backends.cudnn.is_available())
        logger.info(" - cuDNN enabled: %s", torch.backends.cudnn.enabled)
        logger.info(" - allow TF32: %s", torch.backends.cudnn.allow_tf32)
        logger.info(" - deterministic: %s", torch.backends.cudnn.deterministic)
        logger.info(" - benchmark: %s", torch.backends.cudnn.benchmark)
        logger.info(" - benchmark limit: %s", torch.backends.cudnn.benchmark_limit)
    else:
        logger.info("CUDA is not available")
