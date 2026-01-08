import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from torch import nn, optim

from data import find_images, ThermalDataset, analyze_thermal_image
from model import ConvAutoencoder
from train import train_autoencoder, compute_anomaly_scores, percentile_threshold
from viz import visualize_prediction, show_anomaly_map


DATA_DIR = "/content/raw_FLIR_imgs"
OUTPUT_DIR = "./outputs"
BATCH_SIZE = 8
EPOCHS = 30
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUTPUT_DIR, exist_ok=True)

paths = find_images(DATA_DIR)
split = int(0.8 * len(paths))
train_paths, val_paths = paths[:split], paths[split:]

train_loader = DataLoader(ThermalDataset(train_paths), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(ThermalDataset(val_paths), batch_size=BATCH_SIZE)

model = ConvAutoencoder().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

train_autoencoder(model, train_loader, optimizer, criterion, DEVICE, EPOCHS)

scores = compute_anomaly_scores(model, val_loader, DEVICE)
df = pd.DataFrame(scores, columns=["path", "score"])

threshold = percentile_threshold(df["score"].values)
df["prediction"] = df["score"].apply(lambda x: "Anomaly" if x > threshold else "Normal")

df.to_csv(f"{OUTPUT_DIR}/results.csv", index=False)

for _, row in df.head(5).iterrows():
    visualize_prediction(row["path"], row["score"], row["prediction"])

show_anomaly_map(model, val_paths[0], DEVICE)
