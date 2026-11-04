# Copyright (c) 2026 magichan (https://github.com/magichan-lab)

"""縦棒と横棒の使用例を生成する。

パッケージをインストールしてから ``python examples/basic.py`` で実行する。
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

from matplotlib_omission import BrokenBarChart, broken_bar, broken_barh


def main():
    """縦棒・横棒・複数区間を省略したグラフを作成して保存する。"""
    output = Path(__file__).resolve().parent / "output"
    output.mkdir(exist_ok=True)
    labels = ["Product A", "Product B", "Product C", "Product D"]
    values = [1_900_000, 400_000, 16_000_000, 500_000]
    ranges = [(0, 4_500_000), (13_500_000, 16_500_000)]

    vertical = broken_bar(
        labels, values, y_ranges=ranges, figsize=(7, 4.8), color="#5b9bd5", width=0.55
    )
    vertical.set_title("Vertical bars with an omitted Y interval")
    vertical.grid(color="0.85")
    vertical.ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    vertical.figure.subplots_adjust(left=0.18, right=0.96, bottom=0.12, top=0.88)
    vertical.savefig(output / "vertical.png", dpi=160)
    vertical.savefig(output / "vertical.svg")
    vertical.savefig(output / "vertical-transparent.png", dpi=160, transparent=True)

    horizontal = broken_barh(
        labels, values, x_ranges=ranges, figsize=(9, 4.5), color="#5b9bd5", height=0.55
    )
    horizontal.set_title("Horizontal bars with an omitted X interval")
    horizontal.grid(color="0.85")
    horizontal.ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    horizontal.ax.tick_params(axis="x", labelrotation=35, labelsize=8)
    horizontal.figure.subplots_adjust(left=0.13, right=0.97, bottom=0.21, top=0.86)
    horizontal.savefig(output / "horizontal.png", dpi=160)
    horizontal.savefig(output / "horizontal.pdf")

    # 2か所の省略、グループ化した棒、別の省略記号を示す。
    grouped = BrokenBarChart([(0, 5), (18, 25), (90, 105)], break_style="slash", figsize=(7, 5))
    grouped.bar([-0.18, 0.82, 1.82], [3, 22, 97], width=0.36, label="Series A")
    grouped.bar([0.18, 1.18, 2.18], [4, 20, 101], width=0.36, label="Series B")
    grouped.ax.set_xticks([0, 1, 2], ["A", "B", "C"])
    grouped.set_title("Grouped bars with two omitted intervals")
    grouped.grid(color="0.9")
    grouped.legend(loc="upper left")
    grouped.savefig(output / "grouped.png", dpi=160)

    for chart in (vertical, horizontal, grouped):
        plt.close(chart.figure)
    print(f"Examples saved to {output}")


if __name__ == "__main__":
    main()
