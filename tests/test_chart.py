# Copyright (c) 2026 magichan (https://github.com/magichan-lab)

"""省略付き棒グラフの描画、検証、出力形式を確認する。"""

import io

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from matplotlib_omission import BrokenBarChart, broken_bar, broken_barh
from matplotlib_omission._marks import BreakMarks


@pytest.fixture(autouse=True)
def close_figures():
    """各テストの終了後にMatplotlibの図をすべて閉じる。"""
    yield
    plt.close("all")


class TestBrokenBar:
    """縦棒・横棒グラフの表示範囲と省略表現を検証する。"""

    def test_vertical_limits_original_data_categories_and_common_scale(self):
        """縦棒で元データ、カテゴリ、表示区間の共通縮尺を確認する。"""
        chart = broken_bar(["A", "B"], [2, 16], y_ranges=[(0, 4), (14, 17)])
        chart.figure.canvas.draw()
        ax = chart.ax
        assert chart.figure.axes == [ax]
        np.testing.assert_allclose(ax.get_ylim(), (0, 17))
        np.testing.assert_allclose([p.get_height() for p in ax.patches], [2, 16])
        assert [t.get_text() for t in ax.get_xticklabels()] == ["A", "B"]
        lower = ax.transData.transform((0, 2))[1] - ax.transData.transform((0, 1))[1]
        upper = ax.transData.transform((0, 16))[1] - ax.transData.transform((0, 15))[1]
        assert lower == pytest.approx(upper)
        assert not any(4 < tick < 14 for tick in ax.get_yticks())

    def test_horizontal_limits_and_common_scale(self):
        """横棒で表示範囲と各パネルの共通縮尺を確認する。"""
        chart = broken_barh(["A", "B"], [3, 20], x_ranges=[(0, 5), (18, 22)])
        chart.figure.canvas.draw()
        ax = chart.ax
        assert chart.figure.axes == [ax]
        np.testing.assert_allclose(ax.get_xlim(), (0, 22))
        np.testing.assert_allclose([p.get_width() for p in ax.patches], [3, 20])
        lower = ax.transData.transform((2, 0))[0] - ax.transData.transform((1, 0))[0]
        upper = ax.transData.transform((20, 0))[0] - ax.transData.transform((19, 0))[0]
        assert lower == pytest.approx(upper)
        assert not any(5 < tick < 18 for tick in ax.get_xticks())

    def test_wave_clip_is_shared_with_bars_errorbars_and_grid(self):
        """波形のクリップが棒、誤差棒、グリッド線に共有されることを確認する。"""
        chart = broken_bar([0], [16], y_ranges=[(0, 4), (14, 17)], yerr=[0.5])
        chart.grid()
        chart.figure.canvas.draw()
        ax = chart.ax
        assert ax.patches[0].get_clip_path()._patch is ax.patch
        assert ax.collections[0].get_clip_path()._patch is ax.patch
        assert ax.get_ygridlines()[0].get_clip_path()._patch is ax.patch
        path = ax.patch.get_path()
        gap = (ax.patch.bands[0][1] + ax.patch.bands[1][0]) / 2
        assert not path.contains_point((0.5, gap))
        assert path.contains_point((0.5, 0.2))

    @pytest.mark.parametrize("orientation", ["vertical", "horizontal"])
    @pytest.mark.parametrize("gap", [0.08, 0.001])
    def test_wave_boundaries_are_parallel(self, orientation, gap):
        """省略区間の両端が同じ波形になり、間隔が一定であることを確認する。"""
        chart = BrokenBarChart(
            [(0, 4), (14, 18), (28, 32)],
            orientation=orientation,
            gap=gap,
            amplitude=12,
        )
        chart.figure.canvas.draw()
        value = 1 if orientation == "vertical" else 0
        for i in range(2):
            upper = chart.ax.patch.boundaries[i, "high"][:, value]
            lower = chart.ax.patch.boundaries[i + 1, "low"][:, value]
            np.testing.assert_allclose(lower - upper, lower[0] - upper[0], atol=1e-12)
            assert np.ptp(upper) > 0
            np.testing.assert_allclose(upper - upper[0], lower - lower[0], atol=1e-12)

    def test_wave_pixels_are_cut_out_in_transparent_output(self):
        """透過画像で波形の外側だけが切り抜かれることを確認する。"""
        for horizontal in (False, True):
            chart = BrokenBarChart(
                [(0, 4), (14, 18)],
                orientation="horizontal" if horizontal else "vertical",
                amplitude=12,
                wavelength=70,
                figsize=(6, 4),
            )
            (chart.barh if horizontal else chart.bar)([0], [17], color="red")
            chart.figure.canvas.draw()
            ax = chart.ax
            points = ax.patch.boundaries[0, "high"]
            # 棒のカテゴリ範囲にある最も高い山を選ぶ。
            category = 1 if horizontal else 0
            value = 0 if horizontal else 1
            candidates = points[(points[:, category] > 0.3) & (points[:, category] < 0.7)]
            edge = candidates[np.argmax(candidates[:, value])].copy()
            outside, inside = edge.copy(), edge.copy()
            # 表示画素上でアンチエイリアスされた境界線を避ける。
            distance = 6 / (ax.bbox.width if horizontal else ax.bbox.height)
            outside[value] += distance
            inside[value] -= distance
            target = io.BytesIO()
            chart.savefig(target, format="png", transparent=True)
            target.seek(0)
            pixels = plt.imread(target)

            def sample(point):
                """描画領域の座標に対応する画像の画素を返す。"""
                x, y = ax.transAxes.transform(point)
                return pixels[pixels.shape[0] - 1 - int(y), int(x)]

            assert sample(outside)[3] == 0
            assert sample(inside)[3] > 0.9
            assert sample(inside)[0] > 0.9

    def test_wave_amplitude_in_points_survives_resize_and_dpi(self):
        """サイズとDPIを変更しても波形の高さがポイント単位で保たれることを確認する。"""
        chart = broken_bar([0], [16], y_ranges=[(0, 4), (14, 17)])
        for size, dpi in [((6, 4), 100), ((10, 7), 180)]:
            chart.figure.set_size_inches(*size)
            chart.figure.set_dpi(dpi)
            chart.figure.canvas.draw()
            ax = chart.ax
            amplitude = np.ptp(ax.patch.boundaries[0, "high"][:, 1]) * ax.bbox.height / dpi * 72
            assert amplitude == pytest.approx(3, abs=0.02)

    @pytest.mark.parametrize("orientation", ["vertical", "horizontal"])
    def test_bar_slash_marks_only_bars_crossing_break(self, orientation):
        """棒に付く斜線は、その棒がまたぐ省略境界だけに描く。"""
        chart = BrokenBarChart([(0, 4), (14, 18)], orientation=orientation, break_style="bar_slash")
        draw_bar = chart.bar if orientation == "vertical" else chart.barh
        draw_bar([0, 1, 2], [2, 16, 7])
        chart.figure.canvas.draw()
        marks = next(artist for artist in chart.ax.artists if isinstance(artist, BreakMarks))
        dx = chart.figure.dpi / 72 * 4 / chart.ax.bbox.width
        dy = chart.figure.dpi / 72 * 4 / chart.ax.bbox.height
        slashes = marks._bar_slashes(dx, dy)
        assert len(slashes) == 3  # 16は両端、7は下側だけをまたぐ。
        value = 1 if orientation == "vertical" else 0
        boundaries = [slash[:, value].mean() for slash in slashes]
        assert boundaries.count(pytest.approx(chart.ax.patch.bands[0][1])) == 2
        assert boundaries.count(pytest.approx(chart.ax.patch.bands[1][0])) == 1

    def test_grouped_stacked_negative_and_multiple_breaks(self):
        """複数の省略区間とグループ化、積み上げ、負の値を確認する。"""
        chart = BrokenBarChart([(-10, -5), (-2, 4), (15, 22)], break_style="slash")
        chart.bar([0, 1], [-8, 16], width=0.3, label="first")
        series = chart.bar([0, 1], [1, 3], bottom=[-8, 16], width=0.3, label="stack")
        for container in series:
            np.testing.assert_allclose([p.get_y() for p in container], [-8, 16])
        assert len(chart.ax.patches) == 4
        assert len(chart.figure.axes) == 1
        assert len(chart.legend().get_texts()) == 2
        chart.figure.canvas.draw()

    def test_horizontal_stacking_and_errorbars(self):
        """横棒の積み上げと誤差棒を確認する。"""
        chart = BrokenBarChart([(0, 4), (14, 20)], orientation="horizontal")
        chart.barh([0, 1], [2, 16], label="base")
        result = chart.barh([0, 1], [1, 2], left=[2, 16], xerr=0.1)
        for container in result:
            np.testing.assert_allclose([p.get_x() for p in container], [2, 16])
        chart.figure.canvas.draw()

    def test_decimal_ranges_keep_decimal_tick_labels(self):
        """小数の表示範囲で目盛りラベルが小数を保つことを確認する。"""
        chart = broken_bar([0], [0.9], y_ranges=[(0, 0.2), (0.8, 1)])
        chart.figure.canvas.draw()
        assert any("." in t.get_text() for t in chart.axes[0].get_yticklabels())

    def test_custom_ratios_single_interval_and_orientation_guards(self):
        """表示区間比率、単一区間、方向に合わない棒の指定を確認する。"""
        chart = BrokenBarChart([(0, 4), (14, 17)], segment_ratios=[1, 1])
        chart.figure.canvas.draw()
        bands = chart.ax.patch.bands
        assert bands[0][1] - bands[0][0] == pytest.approx(bands[1][1] - bands[1][0])
        with pytest.raises(ValueError):
            chart.barh([0], [16])
        single = broken_barh([0], [2], x_ranges=[(0, 3)])
        assert len(single.axes) == 1
        with pytest.raises(ValueError):
            single.bar([0], [2])

    def test_existing_subplot_grid_uses_one_axes_per_chart(self):
        """既存の2×2配置に縦棒と横棒を各1つのAxesとして描く。"""
        figure, axes = plt.subplots(2, 2)
        vertical = broken_bar(["A"], [16], y_ranges=[(0, 4), (14, 17)], ax=axes[0, 0])
        horizontal = broken_barh(["A"], [16], x_ranges=[(0, 4), (14, 17)], ax=axes[1, 1])
        figure.canvas.draw()
        assert len(figure.axes) == 4
        assert vertical.ax is axes[0, 0]
        assert horizontal.ax is axes[1, 1]
        assert vertical.figure is horizontal.figure is figure

    def test_invalid_configuration_does_not_create_figures(self):
        """無効な設定では例外が発生し図が作成されないことを確認する。"""
        invalid_ranges = [
            [],
            [(0, 0)],
            [(2, 1)],
            [(0, float("inf"))],
            [(0, 4), (3, 5)],
            [(4, 5), (0, 2)],
            [(0, 4), (4, 8)],
            [(0, 1, 2)],
            None,
            [(float("nan"), 2)],
        ]
        for ranges in invalid_ranges:
            with pytest.raises(ValueError):
                BrokenBarChart(ranges)
        for kwargs in [
            dict(orientation="diagonal"),
            dict(break_style="unknown"),
            dict(gap=-1),
            dict(amplitude=-1),
            dict(wavelength=0),
            dict(segment_ratios=[1]),
            dict(segment_ratios=[1, -1]),
            dict(break_linewidth=float("nan")),
            dict(break_color="invalid"),
        ]:
            with pytest.raises(ValueError):
                BrokenBarChart([(0, 4), (14, 17)], **kwargs)
        assert plt.get_fignums() == []

    def test_png_svg_pdf_and_tight_bounds(self):
        """各画像形式と余白を詰めた保存結果を確認する。"""
        for style in ("wave", "slash"):
            chart = broken_bar([0], [16], y_ranges=[(0, 4), (14, 17)], break_style=style)
            for fmt, header in [("png", b"\x89PNG"), ("svg", b"<?xml"), ("pdf", b"%PDF")]:
                output = io.BytesIO()
                chart.savefig(output, format=fmt, bbox_inches="tight", transparent=True, dpi=140)
                assert output.getvalue().startswith(header)
