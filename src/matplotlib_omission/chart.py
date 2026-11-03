# Copyright (c) 2026 magichan (https://github.com/magichan-lab)

"""値軸の一部を省略した棒グラフの公開APIを提供する。

範囲の境界値には、省略による変換を加えていない元の値を使用する。
"""

from collections.abc import Sequence
from math import isfinite

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.ticker import MaxNLocator, ScalarFormatter

from ._marks import BreakMarks, OmissionPatch


def _ranges(ranges):
    """表示範囲を有限な昇順の区間列として検証し、タプルで返す。"""
    try:
        result = tuple(tuple(float(v) for v in interval) for interval in ranges)
    except (TypeError, ValueError) as exc:
        raise ValueError("ranges must contain finite (lower, upper) pairs") from exc
    if not result:
        raise ValueError("at least one visible range is required")
    for i, interval in enumerate(result):
        if len(interval) != 2 or not all(isfinite(v) for v in interval):
            raise ValueError("each range must be a finite (lower, upper) pair")
        low, high = interval
        if low >= high or not isfinite(high - low):
            raise ValueError("each range must have lower < upper and finite span")
        if i and result[i - 1][1] >= low:
            raise ValueError("ranges must be ascending, disjoint, and separated by a gap")
    return result


class BrokenBarChart:
    """値軸の一部を省略した棒グラフの描画領域をまとめる。

    引数
    ----
    ranges : (下限, 上限) の列
        昇順に並べた表示区間。区間の間を省略する。区間が一つの場合は
        省略記号のない通常のグラフを作る。
    orientation : {"vertical", "horizontal"}
        縦棒ではY軸、横棒ではX軸を省略する。
    break_style : {"wave", "slash"}
        各切断辺に波形、または両端に斜線を描く。
    figsize : (幅, 高さ) または None
        Matplotlibの図のサイズ。単位はインチ。
    gap : float
        パネル間隔。パネルの平均サイズに対する比率で指定する。
    segment_ratios : 正の数の列 または None
        昇順の各区間に対応するパネルサイズの比率。省略時は表示区間の幅に
        比例し、すべてのパネルで値軸の縮尺をそろえる。
    amplitude, wavelength : float
        波形の高さと波長。単位はポイントで、出力DPIやサイズ変更に依存しない。
    break_color : 色指定
        省略記号の色。
    break_linewidth : float
        省略記号の線幅。

    補足
    ----
    ``axes`` は縦棒の場合も値の昇順に並ぶ。個々のパネルは通常のMatplotlib
    APIで調整できる。値軸は線形とし、生成後にその範囲、縮尺、順序を
    変更する動作は保証しない。系列を追加するときは ``bar`` または
    ``barh`` を使うと全パネルに反映される。値軸の入力には数値を使う。
    """

    def __init__(
        self,
        ranges: Sequence[tuple[float, float]],
        *,
        orientation="vertical",
        break_style="wave",
        figsize=None,
        gap=0.08,
        segment_ratios=None,
        amplitude=3.0,
        wavelength=36.0,
        break_color="0.25",
        break_linewidth=1.0,
    ):
        """指定した表示区間と省略記号の設定で図と各パネルを作成する。"""
        self.ranges = _ranges(ranges)
        if orientation not in ("vertical", "horizontal"):
            raise ValueError("orientation must be 'vertical' or 'horizontal'")
        if break_style not in ("wave", "slash"):
            raise ValueError("break_style must be 'wave' or 'slash'")
        for name, value, zero_ok in (
            ("gap", gap, True),
            ("amplitude", amplitude, True),
            ("wavelength", wavelength, False),
            ("break_linewidth", break_linewidth, False),
        ):
            if not isfinite(value) or (value < 0 if zero_ok else value <= 0):
                raise ValueError(
                    f"{name} must be finite and {'nonnegative' if zero_ok else 'positive'}"
                )
        to_rgba(break_color)  # 色指定の誤りは図を作成する前に検出する。
        if segment_ratios is None:
            ratios = [high - low for low, high in self.ranges]
        else:
            ratios = list(segment_ratios)
            if len(ratios) != len(self.ranges) or any(not isfinite(r) or r <= 0 for r in ratios):
                raise ValueError("segment_ratios must contain one finite positive value per range")
        # 大きい値でGridSpec内の計算がオーバーフローしないよう正規化する。
        ratios = [r / max(ratios) for r in ratios]
        if any(r == 0 for r in ratios):
            raise ValueError("range sizes differ too much; provide segment_ratios")
        self.orientation = orientation
        self.break_style = break_style
        self.containers = []
        vertical = orientation == "vertical"
        n = len(self.ranges)
        grid = (
            dict(height_ratios=ratios[::-1], hspace=gap)
            if vertical
            else dict(width_ratios=ratios, wspace=gap)
        )
        self.figure, axes = plt.subplots(
            n if vertical else 1,
            1 if vertical else n,
            sharex=vertical,
            sharey=not vertical,
            squeeze=False,
            figsize=figsize,
            gridspec_kw=grid,
        )
        physical_axes = list(axes.flat)
        self.axes = tuple(physical_axes[::-1] if vertical else physical_axes)
        for i, (ax, limits) in enumerate(zip(self.axes, self.ranges)):
            low_cut, high_cut = i > 0, i < n - 1
            if break_style == "wave" and (low_cut or high_cut):
                patch = OmissionPatch(ax, orientation, low_cut, high_cut, amplitude, wavelength)
                patch.set_facecolor(ax.patch.get_facecolor())
                patch.set_edgecolor("none")
                patch.set_figure(self.figure)
                ax.patch = patch
                ax.xaxis.set_clip_path(patch)
                ax.yaxis.set_clip_path(patch)
            value_axis = ax.yaxis if vertical else ax.xaxis
            (ax.set_ylim if vertical else ax.set_xlim)(limits)
            value_axis.set_major_locator(MaxNLocator(nbins="auto", min_n_ticks=2))
            value_axis.set_major_formatter(ScalarFormatter(useOffset=False))
            if vertical:
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                if low_cut:
                    ax.spines["bottom"].set_visible(False)
                    ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
            else:
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                if low_cut:
                    ax.spines["left"].set_visible(False)
                    ax.tick_params(axis="y", which="both", left=False, labelleft=False)
            if low_cut or high_cut:
                ax.add_artist(
                    BreakMarks(
                        ax,
                        orientation,
                        low_cut,
                        high_cut,
                        break_style,
                        break_color,
                        break_linewidth,
                    )
                )

    def bar(self, x, height, width=0.8, bottom=None, **kwargs):
        """すべてのパネルに縦棒の系列を描き、BarContainerのタプルを返す。

        追加のキーワード引数はMatplotlibの ``Axes.bar`` に渡す。
        グループ化には数値のX座標をずらして繰り返し呼び出す。
        積み上げには ``bottom`` を指定する。
        """
        if self.orientation != "vertical":
            raise ValueError("use barh() for a horizontal chart")
        result = tuple(ax.bar(x, height, width=width, bottom=bottom, **kwargs) for ax in self.axes)
        self.containers.append(result)
        return result

    def barh(self, y, width, height=0.8, left=None, **kwargs):
        """すべてのパネルに横棒の系列を描き、BarContainerのタプルを返す。

        追加のキーワード引数は ``Axes.barh`` に渡す。
        グループ化には数値のY座標をずらし、積み上げには ``left`` を使う。
        """
        if self.orientation != "horizontal":
            raise ValueError("use bar() for a vertical chart")
        result = tuple(ax.barh(y, width, height=height, left=left, **kwargs) for ax in self.axes)
        self.containers.append(result)
        return result

    def grid(self, visible=True, *, axis=None, **kwargs):
        """全パネルのグリッド線を設定する。軸の省略時は値軸を使う。"""
        axis = axis or ("y" if self.orientation == "vertical" else "x")
        for ax in self.axes:
            ax.set_axisbelow(True)
            ax.grid(visible, axis=axis, **kwargs)
        return self

    def set_title(self, title, **kwargs):
        """図全体のタイトルを設定し、作成したTextを返す。"""
        return self.figure.suptitle(title, **kwargs)

    def set_xlabel(self, label, **kwargs):
        """図全体のX軸ラベルを設定し、作成したTextを返す。"""
        return self.figure.supxlabel(label, **kwargs)

    def set_ylabel(self, label, **kwargs):
        """図全体のY軸ラベルを設定し、作成したTextを返す。"""
        return self.figure.supylabel(label, **kwargs)

    def legend(self, *args, **kwargs):
        """表示値が最も大きいパネルに凡例を一つ描く。"""
        return self.axes[-1].legend(*args, **kwargs)

    def savefig(self, filename, **kwargs):
        """図を保存する。DPI、透過、ベクター形式などを指定できる。"""
        return self.figure.savefig(filename, **kwargs)

    def show(self, **kwargs):
        """Matplotlibの ``show`` を呼び出して図を表示する。"""
        plt.show(**kwargs)


def broken_bar(
    x,
    height,
    *,
    y_ranges,
    break_style="wave",
    figsize=None,
    gap=0.08,
    segment_ratios=None,
    **bar_kwargs,
):
    """``y_ranges`` の区間間を省略した縦棒グラフを作成する。

    ``bar_kwargs`` は色、棒の幅、基準値、誤差棒、凡例名などの指定として
    ``Axes.bar`` に渡す。詳細な設定には ``BrokenBarChart`` を使用する。
    作成した ``BrokenBarChart`` を返す。
    """
    chart = BrokenBarChart(
        y_ranges, break_style=break_style, figsize=figsize, gap=gap, segment_ratios=segment_ratios
    )
    chart.bar(x, height, **bar_kwargs)
    return chart


def broken_barh(
    y,
    width,
    *,
    x_ranges,
    break_style="wave",
    figsize=None,
    gap=0.08,
    segment_ratios=None,
    **bar_kwargs,
):
    """``x_ranges`` の区間間を省略した横棒グラフを作成する。

    ``bar_kwargs`` は ``Axes.barh`` に渡す。
    作成した ``BrokenBarChart`` を返す。
    """
    chart = BrokenBarChart(
        x_ranges,
        orientation="horizontal",
        break_style=break_style,
        figsize=figsize,
        gap=gap,
        segment_ratios=segment_ratios,
    )
    chart.barh(y, width, **bar_kwargs)
    return chart
