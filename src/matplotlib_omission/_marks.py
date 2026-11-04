# Copyright (c) 2026 magichan (https://github.com/magichan-lab)

"""単一の描画領域で省略区間を切り抜き、境界に省略記号を描く。"""

import numpy as np
from matplotlib.artist import Artist
from matplotlib.container import BarContainer
from matplotlib.lines import Line2D
from matplotlib.patches import PathPatch
from matplotlib.path import Path


class OmissionPatch(PathPatch):
    """表示区間だけを含む背景兼クリップパス。

    ``bands`` は Axes 座標に正規化した表示区間の列とする。
    波形の寸法はポイント単位で保持し、図のサイズやDPIの変更時に更新する。
    """

    def __init__(self, ax, orientation, bands, style, amplitude, wavelength):
        """描画領域と正規化済みの表示区間からクリップパスを作る。"""
        self.owner = ax
        self.orientation = orientation
        self.bands = tuple(bands)
        self.style = style
        self.amplitude = amplitude
        self.wavelength = wavelength
        self._geometry_key = None
        self.boundaries = {}
        super().__init__(Path.unit_rectangle(), transform=ax.transAxes)

    def get_path(self):
        """現在の描画サイズに合わせて各表示区間の境界パスを返す。

        図のサイズまたはDPIが変わった場合はパスと境界座標を再計算する。
        """
        ax = self.owner
        key = (ax.bbox.width, ax.bbox.height, ax.figure.dpi)
        if key != self._geometry_key:
            self._geometry_key = key
            vertical = self.orientation == "vertical"
            along = ax.bbox.width if vertical else ax.bbox.height
            across = ax.bbox.height if vertical else ax.bbox.width
            points_to_pixels = ax.figure.dpi / 72
            cycles = max(along / (self.wavelength * points_to_pixels), 1)
            t = np.linspace(0, 1, max(80, int(cycles * 40)))
            waves = []
            for (band_low, high), (low, band_high) in zip(self.bands, self.bands[1:]):
                amplitude = min(
                    self.amplitude * points_to_pixels / max(across, 1),
                    (high - band_low) * 0.2,
                    (band_high - low) * 0.2,
                    (low - high) * 0.8,
                )
                waves.append(amplitude * (1 - np.cos(2 * np.pi * cycles * t)) / 2)
            vertices, codes = [], []
            self.boundaries = {}
            for i, (low, high) in enumerate(self.bands):
                lower = low + waves[i - 1] if self.style == "wave" and i else np.full_like(t, low)
                upper = (
                    high + waves[i]
                    if self.style == "wave" and i < len(self.bands) - 1
                    else np.full_like(t, high)
                )
                bottom = np.column_stack((t, lower))
                top = np.column_stack((t, upper))
                if not vertical:
                    bottom = bottom[:, ::-1]
                    top = top[:, ::-1]
                self.boundaries[i, "low"] = bottom
                self.boundaries[i, "high"] = top
                segment = np.vstack((bottom, top[::-1], bottom[:1]))
                segment_codes = np.full(len(segment), Path.LINETO, dtype=np.uint8)
                segment_codes[0], segment_codes[-1] = Path.MOVETO, Path.CLOSEPOLY
                vertices.append(segment)
                codes.append(segment_codes)
            self.set_path(Path(np.vstack(vertices), np.concatenate(codes)))
        return super().get_path()


class BreakMarks(Artist):
    """表示区間の境界と値軸の枠線を描画する。

    ``style`` に応じて波形または斜線を描き、棒に付ける斜線には
    元の値軸の表示区間 ``ranges`` を使用する。
    """

    def __init__(self, ax, orientation, patch, ranges, style, color, linewidth):
        """省略記号の描画領域、形状、線の書式を設定する。"""
        super().__init__()
        self.owner = ax
        self.orientation = orientation
        self.patch = patch
        self.ranges = ranges
        self.style = style
        self.color = color
        self.linewidth = linewidth
        self.set_zorder(5)
        self.set_in_layout(False)

    def draw(self, renderer):
        """描画器を使用して表示区間の枠線と省略記号を描く。

        省略記号の形状は ``style`` に従い、不可視のときは描画しない。
        """
        if not self.get_visible():
            return
        ax = self.owner
        self.patch.get_path()
        paths = []
        for low, high in self.patch.bands:
            if self.orientation == "vertical":
                paths.append(np.array([[0, low], [0, high]]))
            else:
                paths.append(np.array([[low, 0], [high, 0]]))
        if self.style == "wave":
            for i in range(len(self.patch.bands) - 1):
                paths.extend(
                    (self.patch.boundaries[i, "high"], self.patch.boundaries[i + 1, "low"])
                )
        else:
            dx = renderer.points_to_pixels(4) / max(ax.bbox.width, 1)
            dy = renderer.points_to_pixels(4) / max(ax.bbox.height, 1)
            ends = (0,) if self.style == "bar_slash" else (0, 1)
            for i in range(len(self.patch.bands) - 1):
                for boundary in (self.patch.bands[i][1], self.patch.bands[i + 1][0]):
                    for end in ends:
                        x, y = (
                            (end, boundary) if self.orientation == "vertical" else (boundary, end)
                        )
                        paths.append(np.array([[x - dx, y - dy], [x + dx, y + dy]]))
            if self.style == "bar_slash":
                paths.extend(self._bar_slashes(dx, dy))
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

    def _bar_slashes(self, dx, dy):
        """省略境界をまたぐ棒の幅に合わせた斜線の座標列を返す。

        ``dx`` と ``dy`` は Axes 座標で表した斜線端のずれとする。
        各棒がまたぐ境界だけに斜線を作る。
        """
        ax = self.owner
        vertical = self.orientation == "vertical"
        limits = ax.get_ylim() if vertical else ax.get_xlim()
        paths = []
        for container in ax.containers:
            if not isinstance(container, BarContainer):
                continue
            for bar in container.patches:
                if not bar.get_visible():
                    continue
                x0, x1 = sorted((bar.get_x(), bar.get_x() + bar.get_width()))
                y0, y1 = sorted((bar.get_y(), bar.get_y() + bar.get_height()))
                value_low, value_high = (y0, y1) if vertical else (x0, x1)
                along_low, along_high = (x0, x1) if vertical else (y0, y1)
                if value_low == value_high or along_low == along_high:
                    continue
                # カテゴリ方向の座標をAxes比率に変換する。
                if vertical:
                    a = ax.transAxes.inverted().transform(
                        ax.transData.transform((along_low, limits[0]))
                    )[0]
                    b = ax.transAxes.inverted().transform(
                        ax.transData.transform((along_high, limits[0]))
                    )[0]
                else:
                    a = ax.transAxes.inverted().transform(
                        ax.transData.transform((limits[0], along_low))
                    )[1]
                    b = ax.transAxes.inverted().transform(
                        ax.transData.transform((limits[0], along_high))
                    )[1]
                a, b = sorted((a, b))
                for i, (left, right) in enumerate(zip(self.patch.bands, self.patch.bands[1:])):
                    edges = ((left[1], self.ranges[i][1]), (right[0], self.ranges[i + 1][0]))
                    for boundary, value in edges:
                        if value_low < value < value_high:
                            if vertical:
                                paths.append(
                                    np.array(
                                        [[a - dx / 2, boundary - dy], [b + dx / 2, boundary + dy]]
                                    )
                                )
                            else:
                                paths.append(
                                    np.array(
                                        [[boundary - dx, a - dy / 2], [boundary + dx, b + dy / 2]]
                                    )
                                )
        return paths
