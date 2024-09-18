from torch import nn
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

def train_step(model: nn.Module,
               train_dataloader: DataLoader,
               loss_fn: nn.Module,
               optimizer: torch.optim,
               device: torch.device):
  model.train()

  train_loss, train_acc = 0, 0

  for batch, (X, y) in enumerate(train_dataloader):
    X, y = X.to(device), y.to(device)
    # Forward pass
    y_logit = model(X)
    # Calculate the loss
    loss = loss_fn(y_logit, y)
    train_loss += loss.item()
    # Optimizer zero grad
    optimizer.zero_grad()
    # loss backward
    loss.backward()
    # Optimizer step
    optimizer.step()
    # Calculate accuracy
    y_label = torch.argmax(torch.softmax(y_logit, dim = 1), dim = 1)
    train_acc += ((y_label == y).sum().item() / len(y_label))
  train_loss /= len(train_dataloader)
  train_acc /= len(train_dataloader)

  return train_loss, train_acc

def test_step(model: nn.Module,
              test_dataloader: DataLoader,
              loss_fn: nn.Module,
              device: torch.device):
  test_loss, test_acc = 0, 0

  model.eval()

  with torch.inference_mode():
    for batch, (X, y) in enumerate(test_dataloader):
      X, y = X.to(device), y.to(device)
      # Forward pass
      y_logit = model(X)
      # Calculate the loss
      loss = loss_fn(y_logit, y)
      test_loss += loss.item()
      # Calculate the acc
      y_label = torch.argmax(torch.softmax(y_logit, dim=1), dim=1)
      test_acc += ((y_label == y).sum().item() / len(y_label))
  test_loss /= len(test_dataloader)
  test_acc /= len(test_dataloader)

  return test_loss, test_acc

def train(model: nn.Module,
          test_dataloader: DataLoader,
          train_dataloader: DataLoader,
          optimizer: torch.optim,
          loss_fn: nn.Module,
          epochs: int,
          device: torch.device):

  results = {
      'epoch' : [],
      'train_loss': [],
      'train_acc' : [],
      'test_loss' : [],
      'test_acc' : []
  }

  for epoch in tqdm(range(epochs)):
    train_loss, train_acc = train_step(model=model,
                                       train_dataloader=train_dataloader,
                                       loss_fn=loss_fn,
                                       optimizer=optimizer,
                                       device=device)

    test_loss, test_acc = test_step(model=model,
                                    test_dataloader=test_dataloader,
                                    loss_fn=loss_fn,
                                    device=device)

    print(
        f'Epoch: {epoch + 1} |'
        f'train_loss: {train_loss:.4f} |'
        f'train_accuracy: {train_acc:.4f} |'
        f'test_loss: {test_loss:.4f} |'
        f'test_accuracy: {test_acc:.4f}'
    )

    results['epoch'].append(epoch + 1)
    results['train_loss'].append(train_loss)
    results['train_acc'].append(train_acc)
    results['test_loss'].append(test_loss)
    results['test_acc'].append(test_acc)

  return results