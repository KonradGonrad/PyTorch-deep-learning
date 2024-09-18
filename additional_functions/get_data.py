from pathlib import Path
import requests
import zipfile
import os

DATA_PATH = Path('data/')
IMAGES_PATH = DATA_PATH / 'pizza_steak_sushi'

if IMAGES_PATH.is_dir():
  print(f'{IMAGES_PATH} already exists.')
else:
  IMAGES_PATH.mkdir(parents=True, exist_ok=True)

  with open(DATA_PATH / 'pizza_steak_sushi.zip', 'wb') as f:
    request = requests.get('https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip')
    f.write(request.content)

  with zipfile.ZipFile(DATA_PATH / 'pizza_steak_sushi.zip', 'r') as zipf:
    zipf.extractall(IMAGES_PATH)

  os.remove(DATA_PATH / 'pizza_steak_sushi.zip')