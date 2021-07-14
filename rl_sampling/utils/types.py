from typing import Dict, List, Union

import torch

TensorLike = Union[
    torch.Tensor, List[torch.Tensor], Dict[str, torch.Tensor],
]
