import torch
from pathlib import Path
from torch import nn

def save_model(model: nn.Module,
               target_dir: str,
               model_name: str):

  target_dir_path = Path(target_dir)
  target_dir_path.mkdir(parents=True,
                        exist_ok=True)

  assert model_name.endswith('.pth') or model_name.endswith('.pt'), 'model_name needs to end with ".pth" or ".ph"'
  model_save_path = target_dir_path / model_name

  torch.save(obj=model.state_dict(),
             f=model_save_path)