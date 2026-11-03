# Copyright (c) 2026 magichan (https://github.com/magichan-lab)

"""値軸を一部省略した縦棒グラフを作成して保存・表示する例。"""

from matplotlib_omission import broken_bar

chart = broken_bar(
    ["A", "B", "C", "D"],
    [1_900_000, 400_000, 16_000_000, 500_000],
    y_ranges=[(0, 4_500_000), (13_500_000, 16_500_000)],
    color="#5b9bd5",
    width=0.55,
    figsize=(7, 5),
)
chart.set_title("Sales")
chart.grid(color="0.85")
chart.savefig("vertical.png", dpi=160, bbox_inches="tight")
chart.show()
