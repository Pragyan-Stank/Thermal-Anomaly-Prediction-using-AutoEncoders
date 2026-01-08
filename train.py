import torch
import pandas as pd
import numpy as np
from typing import List, Tuple
from tqdm import tqdm


def train_autoencoder(model, loader, optimizer, criterion, device, epochs):
    model.to(device)
    model.train()

    for epoch in range(epochs):
        loss_sum = 0
        for imgs, _ in loader:
            imgs = imgs.to(device)
            optimizer.zero_grad()
            recon = model(imgs)
            loss = criterion(recon, imgs)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()

        print(f"Epoch {epoch+1}/{epochs} | Loss: {loss_sum/len(loader):.6f}")


def compute_anomaly_scores(model, loader, device) -> List[Tuple[str, float]]:
    model.eval()
    scores = []

    with torch.no_grad():
        for imgs, paths in loader:
            imgs = imgs.to(device)
            recon = model(imgs)
            errors = torch.mean((imgs - recon) ** 2, dim=[1,2,3]).cpu().numpy()

            for p, e in zip(paths, errors):
                scores.append((p, float(e)))

    return sorted(scores, key=lambda x: x[1], reverse=True)


def percentile_threshold(errors, percentile=95):
    return np.percentile(errors, percentile)
