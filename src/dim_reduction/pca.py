from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.decomposition import PCA


csv_path = Path("../sc2egset.csv").resolve().as_posix()
df = pd.read_csv(csv_path)

X = df['gameloop']
y = df['workersActiveCount']
target_names = df.columns

pca = PCA(n_components=2, random_state=42)

plt.figure()
colors = ["red", "green", "cyan"]
lw = 2

for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(
        x=df['gameloop'], y=df['workersActiveCount'], color=color, alpha=0.1, lw=lw
    )
plt.legend(loc="best", shadow=False, scatterpoints=1)
plt.title("PCA")
plt.xlabel("gameloop")
plt.ylabel("workersActiveCount")
plt.show()