
import abc
from pathlib import Path
import torch
import torch.nn as nn

class BaseModel(nn.Module, abc.ABC):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def forward(self, *args, **kwargs):

        raise NotImplementedError

    def save(self, filepath: str):

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), filepath)

    def load(self, filepath: str, map_location=None):

        state = torch.load(filepath, map_location=map_location)
        self.load_state_dict(state)
        return self

