import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap.umap_ as umap  # Ensure umap-learn is installed
import seaborn as sns

def visualize_2d_embedding(features, labels, method='umap', args=None):
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

    # Map labels to string class names
    label_names = np.array(['anomaly' if lbl == 0 else 'nominal' for lbl in labels])
    palette = {'anomaly': 'red', 'nominal': 'blue'}

    # Plot
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=label_names,
                    palette=palette, s=10, alpha=0.7)
    plt.title(f"{method.upper()} visualization - {args.data}")
    plt.legend(title='Label')
    plt.tight_layout()

    # Save figure
    out_path = f"./figures/tsne/{args.data}_{method}.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved {method} plot to: {out_path}")
