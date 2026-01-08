import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from data import preprocess_image, analyze_thermal_image


def show_anomaly_map(model, img_path, device):
    img = preprocess_image(img_path)
    if img is None:
        return

    tensor = torch.tensor(img).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        recon = model(tensor).cpu().squeeze().numpy()

    diff = np.abs(img - recon)

    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1); plt.imshow(img, cmap='inferno'); plt.title("Original")
    plt.subplot(1,3,2); plt.imshow(recon, cmap='inferno'); plt.title("Reconstruction")
    plt.subplot(1,3,3); plt.imshow(diff, cmap='hot'); plt.title("Anomaly Map")
    plt.show()


def visualize_prediction(img_path, score, label):
    img = cv2.imread(img_path)
    if img is None:
        return

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    stats = analyze_thermal_image(img_path)
    mask = stats.get("mask")

    if mask is not None:
        overlay = img.copy()
        overlay[mask == 255] = [255, 0, 0]
        img = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)

    plt.imshow(img)
    plt.title(f"{label} | Score: {score:.6f}")
    plt.axis("off")
    plt.show()
