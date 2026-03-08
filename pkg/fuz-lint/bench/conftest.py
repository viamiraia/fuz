import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def pytest_benchmark_update_machine_info(config, machine_info):
    """Add GPU information to benchmark machine_info."""
    import torch

    if torch.cuda.is_available():
        machine_info["gpu"] = {
            "name": torch.cuda.get_device_name(0),
            "count": torch.cuda.device_count(),
            "memory_mb": torch.cuda.get_device_properties(0).total_memory
            // (1024 * 1024),
        }
    else:
        machine_info["gpu"] = None
