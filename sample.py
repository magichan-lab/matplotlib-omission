# Copyright (c) 2026 magichan (https://github.com/magichan-lab)

"""Matplotlib 統合形式で値軸を一部省略した縦棒グラフを作成する例。"""

import matplotlib.pyplot as plt

import matplotlib_omission

matplotlib_omission.enable()

fig, ax = plt.subplots(figsize=(7, 5))
ax.bar(
    ["A", "B", "C", "D"],
    [1_900_000, 400_000, 16_000_000, 500_000],
    color="#5b9bd5",
    width=0.55,
)
ax.set_ylim(0, 16_500_000, step=1_500_000, omit=(4_500_000, 13_500_000))
ax.set_title("Sales")
ax.grid(axis="y", color="0.85")
fig.savefig("vertical.png", dpi=160, bbox_inches="tight")
plt.show()
