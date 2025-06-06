import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap.umap_ as umap  # Ensure umap-learn is installed
import seaborn as sns

def visualize_2d_embedding(features, labels, method='umap', args=None):
    ref_font_size = 22
    os.makedirs(f"./figures/tsne/", exist_ok=True)

    # Choose dimensionality reduction method
    if method == 'umap':
        reducer = umap.UMAP(n_components=2, random_state=42)
    elif method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
    else:
        raise ValueError("Method must be 'umap' or 'tsne'.")

    # Fit and transform features
    embedding = reducer.fit_transform(features)

    # Prepare data for plotting
    embedding = np.array(embedding)
    labels = np.array(labels)

    # Separate indices
    anomaly_idx = labels == 0
    nominal_idx = labels == 1

    plt.figure(figsize=(12, 8))

    # Plot nominal (blue circles)
    plt.scatter(embedding[nominal_idx, 0], embedding[nominal_idx, 1],
                c='blue', label='nominal', marker='o', s=60, alpha=0.6, edgecolor='black')

    # Plot anomaly (red triangles)
    plt.scatter(embedding[anomaly_idx, 0], embedding[anomaly_idx, 1],
                c='red', label='anomaly', marker='^', s=80, alpha=0.7, edgecolor='black')

    # Final formatting
    plt.title(f"{method.upper()} visualization - {args.data}", fontsize=ref_font_size+4)
    plt.legend(fontsize=ref_font_size-2, loc='upper right')
    plt.xlabel('Component 1', fontsize=ref_font_size)
    plt.ylabel('Component 2', fontsize=ref_font_size)
    plt.tight_layout()

    # Save figure
    out_path = f"./figures/tsne/{args.data}_{method}.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved {method} plot to: {out_path}")

