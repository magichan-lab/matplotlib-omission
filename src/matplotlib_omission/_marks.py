# Copyright (c) 2026 magichan (https://github.com/magichan-lab)

"""描画領域の切り抜きと省略記号に共通する境界線を定義する。"""

import numpy as np
from matplotlib.artist import Artist
from matplotlib.lines import Line2D
from matplotlib.patches import PathPatch
from matplotlib.path import Path


class OmissionPatch(PathPatch):
    """省略位置に内向きの波形境界を持つ描画領域の背景兼クリップパス。

    描画領域のサイズまたは出力DPIが変わるとパスを再計算する。
    描画領域のパッチをクリップに使うため、透過出力や誤差棒にも適用される。
    """

    def __init__(self, ax, orientation, low_cut, high_cut, amplitude, wavelength):
        """描画領域、切断位置、波形の寸法からパッチを初期化する。"""
        self.owner = ax
        self.orientation = orientation
        self.low_cut = low_cut
        self.high_cut = high_cut
        self.amplitude = amplitude
        self.wavelength = wavelength
        self._geometry_key = None
        self.boundaries = {}
        super().__init__(Path.unit_rectangle(), transform=ax.transAxes)

    def get_path(self):
        """現在の描画サイズに合わせて境界パスを更新して返す。"""
        ax = self.owner
        key = (ax.bbox.width, ax.bbox.height, ax.figure.dpi)
        if key != self._geometry_key:
            self._geometry_key = key
            vertical = self.orientation == "vertical"
            along = ax.bbox.width if vertical else ax.bbox.height
            across = ax.bbox.height if vertical else ax.bbox.width
            points_to_pixels = ax.figure.dpi / 72
            # サイズ変更後も非常に短いパネルを描画できるようにする。
            amplitude = min(self.amplitude * points_to_pixels / max(across, 1), 0.12)
            cycles = max(along / (self.wavelength * points_to_pixels), 1)
            t = np.linspace(0, 1, max(80, int(cycles * 40)))
            wave = amplitude * (1 - np.cos(2 * np.pi * cycles * t)) / 2
            low = wave if self.low_cut else np.zeros_like(t)
            high = 1 - wave if self.high_cut else np.ones_like(t)
            bottom = np.column_stack((t, low))
            top = np.column_stack((t, high))
            if not vertical:
                bottom = bottom[:, ::-1]
                top = top[:, ::-1]
            self.boundaries = {"low": bottom, "high": top}
            vertices = np.vstack((bottom, top[::-1], bottom[:1]))
            codes = np.full(len(vertices), Path.LINETO, dtype=np.uint8)
            codes[0], codes[-1] = Path.MOVETO, Path.CLOSEPOLY
            self.set_path(Path(vertices, codes))
        return super().get_path()


class BreakMarks(Artist):
    """パネルの切断辺に波形または斜線の省略記号を描画する。"""

    def __init__(self, ax, orientation, low_cut, high_cut, style, color, linewidth):
        """描画領域と省略記号の表示条件・書式を設定する。"""
        super().__init__()
        self.owner = ax
        self.orientation = orientation
        self.cuts = {"low": low_cut, "high": high_cut}
        self.style = style
        self.color = color
        self.linewidth = linewidth
        self.set_zorder(5)
        self.set_in_layout(False)

    def draw(self, renderer):
        """有効な切断辺の省略記号を指定されたレンダラーで描画する。"""
        if not self.get_visible():
            return
        ax = self.owner
        if self.style == "wave":
            ax.patch.get_path()
            paths = [ax.patch.boundaries[edge] for edge, cut in self.cuts.items() if cut]
        else:
            # 斜線の長さは描画領域のサイズに依存しないポイント単位で指定する。
            dx = renderer.points_to_pixels(4) / max(ax.bbox.width, 1)
            dy = renderer.points_to_pixels(4) / max(ax.bbox.height, 1)
            paths = []
            for edge, cut in self.cuts.items():
                if not cut:
                    continue
                boundary = 0 if edge == "low" else 1
                for end in (0, 1):
                    x, y = (end, boundary) if self.orientation == "vertical" else (boundary, end)
                    paths.append(np.array([[x - dx, y - dy], [x + dx, y + dy]]))
        for vertices in paths:
            line = Line2D(
                vertices[:, 0],
                vertices[:, 1],
                transform=ax.transAxes,
                color=self.color,
                linewidth=self.linewidth,
                clip_on=False,
            )
            line.set_figure(ax.figure)
            line.draw(renderer)
        self.stale = False
