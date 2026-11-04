# Copyright (c) 2026 magichan (https://github.com/magichan-lab)

"""Matplotlib の軸範囲設定に省略区間の指定を追加する。"""

from functools import wraps

import numpy as np
from matplotlib.axes import Axes

from .chart import BrokenBarChart

_OPTIONS = {  # 省略区間の設定時に受け付ける追加オプション
    "break_style",
    "gap",
    "segment_ratios",
    "amplitude",
    "wavelength",
    "break_color",
    "break_linewidth",
}
_enabled = False  # Matplotlib のメソッドを拡張済みかを示す。


def _is_ranges(value):
    """値が表示区間の列として渡されたかを判定する。

    数値の上下限の組は通常の Matplotlib の指定として扱う。
    空の列は省略区間の指定とみなし、区間の検証時に拒否する。
    """
    if not isinstance(value, (list, tuple, np.ndarray)):
        return False
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return False
    return len(value) == 0 or isinstance(value[0], (list, tuple, np.ndarray))


def _wrap_limit(original, orientation):
    """通常の数値範囲を維持しつつ、区間列を省略グラフへ変換する関数を返す。

    ``orientation`` は省略する値軸の向きであり、横向きなら X 軸、
    縦向きなら Y 軸を対象とする。
    """

    @wraps(original, assigned=("__module__", "__name__", "__qualname__", "__annotations__"))
    def set_limit(self, *args, **kwargs):
        """軸の表示範囲を設定し、設定後の上下限を返す。

        数値の上下限は Matplotlib 本来の範囲設定に渡す。
        ``[(下限, 上限), ...]`` を渡すと、列に含まれる区間だけを表示する。
        省略区間には ``break_style``、``gap``、``segment_ratios``、
        ``amplitude``、``wavelength``、``break_color``、
        ``break_linewidth`` を追加で指定できる。
        不正な引数には ``TypeError``、重複した省略軸には ``ValueError`` を送出する。
        """
        lower_name, upper_name = (
            ("left", "right") if orientation == "horizontal" else ("bottom", "top")
        )
        lower = args[0] if args else kwargs.get(lower_name)
        if not _is_ranges(lower):
            return original(self, *args, **kwargs)
        aliases = ("xmin", "xmax") if orientation == "horizontal" else ("ymin", "ymax")
        if len(args) > 1 or upper_name in kwargs or any(name in kwargs for name in aliases):
            raise TypeError("visible ranges must be the only limit argument")
        if args and lower_name in kwargs:
            raise TypeError("visible ranges were specified more than once")
        if kwargs.get("emit", True) is not True or kwargs.get("auto", False) is not False:
            raise TypeError("emit and auto cannot be used with visible ranges")
        invalid = kwargs.keys() - _OPTIONS - {lower_name, "emit", "auto"}
        if invalid:
            raise TypeError(f"unexpected option for visible ranges: {sorted(invalid)[0]}")
        if getattr(self, "_matplotlib_omission_chart", None) is not None:
            raise ValueError("this Axes already has an omitted value axis")
        options = {name: kwargs[name] for name in _OPTIONS if name in kwargs}
        chart = BrokenBarChart(lower, orientation=orientation, ax=self, **options)
        self._matplotlib_omission_chart = chart
        return self.get_xlim() if orientation == "horizontal" else self.get_ylim()

    return set_limit


def enable():
    """Matplotlib の ``set_xlim`` / ``set_ylim`` で省略区間を指定可能にする。

    ``ax.set_ylim([(0, 4), (14, 18)])`` のように表示区間を昇順で渡す。
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
