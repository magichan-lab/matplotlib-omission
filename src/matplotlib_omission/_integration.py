# Copyright (c) 2026 magichan (https://github.com/magichan-lab)

"""Matplotlib の軸範囲設定に省略区間の指定を追加する。"""

from functools import wraps
from inspect import signature
from math import floor, isfinite

import numpy as np
from matplotlib.axes import Axes
from matplotlib.ticker import FixedLocator

from .chart import BrokenBarChart, _ranges

_OPTIONS = {  # 省略区間の設定時に受け付ける追加オプション
    "break_style",
    "gap",
    "segment_ratios",
    "amplitude",
    "wavelength",
    "break_color",
    "break_linewidth",
    "show_break_tick_labels",
}
_enabled = False  # Matplotlib のメソッドを拡張済みかを示す。


def _omitted_ranges(omit, lower, upper):
    """省略区間を検証し、上下限の内側にある区間のタプルを返す。"""
    if (
        isinstance(omit, (list, tuple, np.ndarray))
        and len(omit) == 2
        and all(np.isscalar(value) for value in omit)
    ):
        omit = (omit,)
    ranges = _ranges(omit)
    if ranges[0][0] <= lower or ranges[-1][1] >= upper:
        raise ValueError("omitted ranges must lie strictly inside the axis limits")
    return ranges


def _visible_ranges(lower, upper, omitted):
    """全体の上下限から省略区間を引いた表示区間を返す。"""
    edges = [lower]
    for start, end in omitted:
        edges.extend((start, end))
    edges.append(upper)
    return tuple((edges[i], edges[i + 1]) for i in range(0, len(edges), 2))


def _step_ticks(lower, upper, step, omitted=()):
    """一定間隔で目盛りを生成し、省略区間内の値を除外して返す。"""
    try:
        step = float(step)
    except (TypeError, ValueError) as exc:
        raise ValueError("step must be a finite positive number") from exc
    if not isfinite(step) or step <= 0:
        raise ValueError("step must be a finite positive number")
    count = (upper - lower) / step
    if not isfinite(count) or count > 10_000:
        raise ValueError("step creates too many ticks")
    ticks = lower + np.arange(floor(count) + 1) * step
    return [
        tick
        for tick in ticks
        if tick <= upper and not any(start < tick < end for start, end in omitted)
    ]


def _set_step(ax, orientation, lower, upper, step, omitted=()):
    """対象の軸に一定間隔の目盛りを設定する。"""
    ticks = _step_ticks(lower, upper, step, omitted)
    axis = ax.xaxis if orientation == "horizontal" else ax.yaxis
    axis.set_major_locator(FixedLocator(ticks))


def _wrap_limit(original, orientation):
    """通常の上下限に省略区間と目盛り間隔を追加した関数を返す。

    ``orientation`` は省略する値軸の向きであり、横向きなら X 軸、
    縦向きなら Y 軸を対象とする。
    """

    @wraps(original, assigned=("__module__", "__name__", "__qualname__", "__annotations__"))
    def set_limit(self, *args, **kwargs):
        """軸の表示範囲を設定し、設定後の上下限を返す。

        数値の上下限に ``omit=(開始, 終了)`` を追加するとその区間を省略する。
        ``omit`` には複数の区間も指定できる。``step`` は目盛り間隔を指定する。
        省略軸には ``break_style``、``gap``、``segment_ratios``、
        ``amplitude``、``wavelength``、``break_color``、
        ``break_linewidth``、``show_break_tick_labels`` を追加で指定できる。
        省略軸の再設定や不正な区間には ``ValueError`` を送出する。
        """
        omit = kwargs.pop("omit", None)
        step = kwargs.pop("step", None)
        if omit is None and step is None:
            return original(self, *args, **kwargs)
        options = {name: kwargs.pop(name) for name in _OPTIONS if name in kwargs}
        if omit is None and options:
            raise TypeError("break options require omit")
        lower_name, upper_name = (
            ("left", "right") if orientation == "horizontal" else ("bottom", "top")
        )
        bound = signature(original).bind(self, *args, **kwargs).arguments
        aliases = ("xmin", "xmax") if orientation == "horizontal" else ("ymin", "ymax")
        if lower_name in bound and aliases[0] in bound:
            raise TypeError("lower limit was specified more than once")
        if upper_name in bound and aliases[1] in bound:
            raise TypeError("upper limit was specified more than once")
        lower = bound.get(lower_name, bound.get(aliases[0]))
        upper = bound.get(upper_name, bound.get(aliases[1]))
        if upper is None and isinstance(lower, (list, tuple, np.ndarray)) and len(lower) == 2:
            lower, upper = lower
        if omit is None:
            limits = original(self, *args, **kwargs)
            _set_step(self, orientation, *sorted(limits), step)
            return limits
        if kwargs.get("emit", True) is not True or kwargs.get("auto", False) is not False:
            raise TypeError("emit and auto cannot be used with omit")
        if lower is None or upper is None:
            raise ValueError("both axis limits are required with omit")
        try:
            lower, upper = float(lower), float(upper)
        except (TypeError, ValueError) as exc:
            raise ValueError("axis limits must be finite numbers") from exc
        if not isfinite(lower) or not isfinite(upper) or lower >= upper:
            raise ValueError("axis limits must be finite and ascending")
        omitted = _omitted_ranges(omit, lower, upper)
        if step is not None:
            # 目盛り数の誤りは軸を変更する前に検出する。
            _step_ticks(lower, upper, step, omitted)
        if getattr(self, "_matplotlib_omission_chart", None) is not None:
            raise ValueError("this Axes already has an omitted value axis")
        chart = BrokenBarChart(
            _visible_ranges(lower, upper, omitted), orientation=orientation, ax=self, **options
        )
        self._matplotlib_omission_chart = chart
        if step is not None:
            _set_step(self, orientation, lower, upper, step, omitted)
        return self.get_xlim() if orientation == "horizontal" else self.get_ylim()

    return set_limit


def enable():
    """Matplotlib の ``set_xlim`` / ``set_ylim`` で省略区間を指定可能にする。

    ``ax.set_ylim(0, 18, step=2, omit=(4, 14))`` のように指定する。
    ``plt.xlim`` / ``plt.ylim`` でも同じ形式を使用できる。繰り返し呼んでも
    ラッパーは重複しない。この変更はプロセス内のすべての ``Axes`` に適用する。
    戻り値はない。
    """
    global _enabled
    if _enabled:
        return
    Axes.set_xlim = _wrap_limit(Axes.set_xlim, "horizontal")
    Axes.set_ylim = _wrap_limit(Axes.set_ylim, "vertical")
    _enabled = True
