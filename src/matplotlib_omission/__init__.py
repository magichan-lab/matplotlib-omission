# Copyright (c) 2026 magichan (https://github.com/magichan-lab)

"""縦棒・横棒グラフの値軸の一部を省略するグラフを提供する。"""

from ._integration import enable
from .chart import BrokenBarChart, broken_bar, broken_barh

__all__ = ["BrokenBarChart", "broken_bar", "broken_barh", "enable"]  # 公開API
__version__ = "0.1.0"  # パッケージのバージョン
