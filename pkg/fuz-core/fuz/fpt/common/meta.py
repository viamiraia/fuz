from typing import TypeAlias

import torch
from attrs import frozen

DevType: TypeAlias = str | torch.device


@frozen
class TorchCfg:
    device: DevType
    seed: int | None = None


def get_cuda_info() -> list[str]:
    if torch.cuda.is_available():
        return [
            torch.cuda.get_device_properties(i).name
            for i in range(torch.cuda.device_count())
        ]
    return []


def auto_setup(seed: int | None = None) -> TorchCfg:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_default_device(device)
    return TorchCfg(device=device, seed=seed)


def handle_tcfg(tcfg: TorchCfg | None = None):
    if tcfg is None:
        return torch.get_default_device()
    device = torch.get_default_device() if tcfg.device is None else tcfg.device
    if tcfg.seed is not None and torch.seed() != tcfg.seed:
        torch.manual_seed(tcfg.seed)
    return device
