# Copyright (c) 2026 magichan (https://github.com/magichan-lab)

"""値軸の一部を省略した棒グラフの公開APIを提供する。

範囲の境界値には、省略による変換を加えていない元の値を使用する。
"""

from collections.abc import Sequence
from math import isfinite

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.ticker import FixedLocator, MaxNLocator, ScalarFormatter

from ._marks import BreakMarks, OmissionPatch


def _ranges(ranges):
    """表示範囲を検証し、数値に変換した区間のタプルを返す。

    区間は有限な下限と上限を持ち、昇順で互いに離れている必要がある。
    不正な区間がある場合は ``ValueError`` を送出する。
    """
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


def _piecewise(values, source, target):
    """値を節点間で線形補間し、節点の外側では端の傾きで外挿する。

    ``source`` と ``target`` は対応する節点の座標列とする。
    変換した値を NumPy 配列で返す。
    """
    values = np.asarray(values, dtype=float)
    result = np.interp(values, source, target)
    lower_slope = (target[1] - target[0]) / (source[1] - source[0])
    upper_slope = (target[-1] - target[-2]) / (source[-1] - source[-2])
    return np.where(
        values < source[0],
        target[0] + (values - source[0]) * lower_slope,
        np.where(values > source[-1], target[-1] + (values - source[-1]) * upper_slope, result),
    )


class BrokenBarChart:
    """単一の描画領域で値軸の一部を省略した棒グラフを作る。

    引数
    ----
    ranges : (下限, 上限) の列
        昇順に並べた表示区間。区間の間を省略する。区間が一つの場合は
        省略記号のない通常のグラフを作る。
    orientation : {"vertical", "horizontal"}
        縦棒ではY軸、横棒ではX軸を省略する。
    break_style : {"wave", "slash", "bar_slash"}
        各切断辺に波形、値軸のみに斜線、または値軸と棒に斜線を描く。
    figsize : (幅, 高さ) または None
        新しい図のサイズ。単位はインチ。``ax`` 指定時は使用できない。
    ax : matplotlib.axes.Axes または None
        描画先。省略時は新しい図とAxesを作る。
    gap : float
        省略区間の画面上の間隔。表示区間の平均サイズに対する比率。
    segment_ratios : 正の数の列 または None
        昇順の各表示区間に対応する画面上の長さの比率。省略時は区間の幅に
        比例し、各区間で値軸の縮尺をそろえる。
    amplitude, wavelength : float
        波形の高さと波長。単位はポイントで、出力DPIやサイズ変更に依存しない。
    break_color : 色指定
        省略記号の色。
    break_linewidth : float
        省略記号の線幅。

    補足
    ----
    ``ax`` が唯一のAxesで、``axes`` は互換性のためそれを含むタプルとする。
    値軸には区分線形のスケールを設定するため、生成後にその範囲や縮尺を
    変更する動作は保証しない。系列の追加には ``bar`` または ``barh`` を使う。
    値軸の入力には数値を使う。
    """

    def __init__(
        self,
        ranges: Sequence[tuple[float, float]],
        *,
        orientation="vertical",
        break_style="wave",
        figsize=None,
        ax=None,
        gap=0.08,
        segment_ratios=None,
        amplitude=3.0,
        wavelength=36.0,
        break_color="0.25",
        break_linewidth=1.0,
    ):
        """指定した表示区間と省略記号の設定で単一のAxesを構成する。"""
        self.ranges = _ranges(ranges)
        if orientation not in ("vertical", "horizontal"):
            raise ValueError("orientation must be 'vertical' or 'horizontal'")
        if break_style not in ("wave", "slash", "bar_slash"):
            raise ValueError("break_style must be 'wave', 'slash', or 'bar_slash'")
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
        if ax is not None and figsize is not None:
            raise ValueError("figsize cannot be used with ax")
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
        if ax is None:
            self.figure, self.ax = plt.subplots(figsize=figsize)
        else:
            self.ax = ax
            self.figure = ax.figure
        self.axes = (self.ax,)
        positions = [0.0]
        values = [self.ranges[0][0]]
        for i, (low, high) in enumerate(self.ranges):
            if i:
                positions.append(positions[-1] + gap * sum(ratios) / n)
                values.append(low)
            positions.append(positions[-1] + ratios[i])
            values.append(high)
        scale = positions[-1]
        bands = tuple((positions[2 * i] / scale, positions[2 * i + 1] / scale) for i in range(n))
        value_axis = self.ax.yaxis if vertical else self.ax.xaxis
        (self.ax.set_yscale if vertical else self.ax.set_xscale)(
            "function",
            functions=(
                lambda x: _piecewise(x, values, positions),
                lambda x: _piecewise(x, positions, values),
            ),
        )
        (self.ax.set_ylim if vertical else self.ax.set_xlim)(values[0], values[-1])
        (self.ax.set_autoscaley_on if vertical else self.ax.set_autoscalex_on)(False)
        ticks = []
        for low, high in self.ranges:
            candidates = MaxNLocator(nbins=4, min_n_ticks=2).tick_values(low, high)
            ticks.extend(t for t in candidates if low <= t <= high)
        value_axis.set_major_locator(FixedLocator(sorted(set(ticks))))
        value_axis.set_major_formatter(ScalarFormatter(useOffset=False))
        if n > 1:
            patch = OmissionPatch(self.ax, orientation, bands, break_style, amplitude, wavelength)
            patch.set_facecolor(self.ax.patch.get_facecolor())
            patch.set_edgecolor("none")
            patch.set_figure(self.figure)
            self.ax.patch = patch
            # 軸範囲を設定する前に描いた系列にも省略区間のクリップを適用する。
            for artist in (*self.ax.patches, *self.ax.collections, *self.ax.lines):
                if artist is not patch and artist.get_clip_on() and artist.get_clip_path() is None:
                    artist.set_clip_path(patch)
            self.ax.xaxis.set_clip_path(patch)
            self.ax.yaxis.set_clip_path(patch)
            self.ax.add_artist(
                BreakMarks(
                    self.ax,
                    orientation,
                    patch,
                    self.ranges,
                    break_style,
                    break_color,
                    break_linewidth,
                )
            )
            self.ax.spines["left" if vertical else "bottom"].set_visible(False)
        self.ax.spines["top"].set_visible(False)
        self.ax.spines["right"].set_visible(False)

    def bar(self, x, height, width=0.8, bottom=None, **kwargs):
        """単一のAxesに縦棒の系列を描き、BarContainerのタプルを返す。

        ``x`` は棒の位置、``height`` は棒の高さ、``width`` は棒の幅を指定する。
        ``bottom`` は各棒の開始位置を指定し、積み上げに利用できる。
        追加のキーワード引数はMatplotlibの ``Axes.bar`` に渡す。
        グループ化には数値のX座標をずらして繰り返し呼び出す。
        グラフが横向きの場合は ``ValueError`` を送出する。
        """
        if self.orientation != "vertical":
            raise ValueError("use barh() for a horizontal chart")
        result = (self.ax.bar(x, height, width=width, bottom=bottom, **kwargs),)
        self.containers.append(result)
        return result

    def barh(self, y, width, height=0.8, left=None, **kwargs):
        """単一のAxesに横棒の系列を描き、BarContainerのタプルを返す。

        ``y`` は棒の位置、``width`` は棒の長さ、``height`` は棒の太さを指定する。
        ``left`` は各棒の開始位置を指定し、積み上げに利用できる。
        追加のキーワード引数は ``Axes.barh`` に渡す。
        グループ化には数値のY座標をずらして繰り返し呼び出す。
        グラフが縦向きの場合は ``ValueError`` を送出する。
        """
        if self.orientation != "horizontal":
            raise ValueError("use bar() for a vertical chart")
        result = (self.ax.barh(y, width, height=height, left=left, **kwargs),)
        self.containers.append(result)
        return result

    def grid(self, visible=True, *, axis=None, **kwargs):
        """グリッド線を設定し、自身を返す。

        ``visible`` で表示を切り替える。``axis`` の省略時は値軸を使い、
        追加のキーワード引数は ``Axes.grid`` に渡す。
        """
        axis = axis or ("y" if self.orientation == "vertical" else "x")
        self.ax.set_axisbelow(True)
        self.ax.grid(visible, axis=axis, **kwargs)
        return self

    def set_title(self, title, **kwargs):
        """タイトルを設定し、作成した ``Text`` を返す。

        追加のキーワード引数は ``Axes.set_title`` に渡す。
        """
        return self.ax.set_title(title, **kwargs)

    def set_xlabel(self, label, **kwargs):
        """X軸ラベルを設定し、作成した ``Text`` を返す。

        追加のキーワード引数は ``Axes.set_xlabel`` に渡す。
        """
        return self.ax.set_xlabel(label, **kwargs)

    def set_ylabel(self, label, **kwargs):
        """Y軸ラベルを設定し、作成した ``Text`` を返す。

        追加のキーワード引数は ``Axes.set_ylabel`` に渡す。
        """
        return self.ax.set_ylabel(label, **kwargs)

    def legend(self, *args, **kwargs):
        """凡例を描き、作成した ``Legend`` を返す。

        引数は ``Axes.legend`` に渡す。
        """
        return self.ax.legend(*args, **kwargs)

    def savefig(self, filename, **kwargs):
        """図をファイル名またはファイルオブジェクトに保存する。

        ``filename`` は保存先を指定する。DPI、透過、出力形式などの
        キーワード引数は ``Figure.savefig`` に渡す。
        """
        return self.figure.savefig(filename, **kwargs)

    def show(self, **kwargs):
        """Matplotlib の ``show`` を呼び出して図を表示する。

        キーワード引数は ``pyplot.show`` に渡す。
        """
        plt.show(**kwargs)


def broken_bar(
    x,
    height,
    *,
    y_ranges,
    break_style="wave",
    figsize=None,
    ax=None,
    gap=0.08,
    segment_ratios=None,
    **bar_kwargs,
):
    """``y_ranges`` の区間間を省略した縦棒グラフを作成する。

    ``x`` は棒の位置、``height`` は棒の高さを指定する。
    ``y_ranges`` はY軸に表示する区間を昇順で指定する。
    ``ax`` には既存のAxesを指定できる。省略時は新しい図を作る。
    ``break_style``、``figsize``、``gap``、``segment_ratios`` は
    ``BrokenBarChart`` と同じ意味で使用する。
    ``bar_kwargs`` は色、棒の幅、基準値、誤差棒、凡例名などの指定として
    ``Axes.bar`` に渡す。詳細な設定には ``BrokenBarChart`` を使用する。
    作成した ``BrokenBarChart`` を返す。
    """
    chart = BrokenBarChart(
        y_ranges,
        break_style=break_style,
        figsize=figsize,
        ax=ax,
        gap=gap,
        segment_ratios=segment_ratios,
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
    ax=None,
    gap=0.08,
    segment_ratios=None,
    **bar_kwargs,
):
    """``x_ranges`` の区間間を省略した横棒グラフを作成する。

    ``y`` は棒の位置、``width`` は棒の長さを指定する。
    ``x_ranges`` はX軸に表示する区間を昇順で指定する。
    ``ax`` には既存のAxesを指定できる。省略時は新しい図を作る。
    ``break_style``、``figsize``、``gap``、``segment_ratios`` は
    ``BrokenBarChart`` と同じ意味で使用する。
    ``bar_kwargs`` は ``Axes.barh`` に渡す。
    作成した ``BrokenBarChart`` を返す。
    """
    chart = BrokenBarChart(
        x_ranges,
        orientation="horizontal",
        break_style=break_style,
        figsize=figsize,
        ax=ax,
        gap=gap,
        segment_ratios=segment_ratios,
    )
    chart.barh(y, width, **bar_kwargs)
    return chart
