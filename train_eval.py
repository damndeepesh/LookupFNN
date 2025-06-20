import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score
import time

def train(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)

def evaluate(model, loader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            pred = torch.argmax(logits, dim=1).cpu().numpy()
            preds.extend(pred)
            labels.extend(y.cpu().numpy())
    return accuracy_score(labels, preds)

def benchmark_inference(model, loader, device, n_repeats=3):
    model.eval()
    times = []
    with torch.no_grad():
        for _ in range(n_repeats):
            start = time.time()
            for x, _ in loader:
                x = x.to(device)
                _ = model(x)
            times.append(time.time() - start)
    return np.mean(times) 