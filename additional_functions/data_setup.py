from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from get_data import IMAGES_PATH

TRAIN_DIR = IMAGES_PATH / 'train'
TEST_DIR = IMAGES_PATH / 'test'
BATCH_SIZE = 1

simple_transform = transforms.Compose([
    transforms.Resize(size=(64, 64)),
    transforms.ToTensor()
])

def prepare_data(transforms: transforms = simple_transform,
                 train_dir: str = TRAIN_DIR,
                 test_dir: str = TEST_DIR,
                 batch_size: int = BATCH_SIZE
                 ):
  """
    Function that prepares data and transform it into dict of train_data, test_data, train_dataloader, test_dataloader and class_names.
    Returns dict
  """
  prepared_data = {
      'class_names': None,
      'train_data' : None,
      'test_data' : None,
      'train_dataloader' : None,
      'test_dataloader' : None
  }

  train_data = datasets.ImageFolder(root=train_dir,
                                    transform = transforms,
                                    target_transform= None)

  test_data = datasets.ImageFolder(root=test_dir,
                                   transform=transforms,
                                   target_transform=None)

  class_names = train_data.classes

  train_dataloader = DataLoader(train_data,
                                batch_size=batch_size,
                                shuffle=True)

  test_dataloader = DataLoader(test_data,
                               batch_size=batch_size,
                               shuffle=False)

  prepared_data['class_names'] = class_names
  prepared_data['train_data'] = train_data
  prepared_data['test_data'] = test_data
  prepared_data['train_dataloader'] = train_dataloader
  prepared_data['test_dataloader'] = test_dataloader

  return prepared_data