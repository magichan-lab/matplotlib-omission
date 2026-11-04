# Copyright (c) 2026 magichan (https://github.com/magichan-lab)

"""Matplotlib の軸範囲 API から省略区間を指定する動作を検証する。"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

import matplotlib_omission
from matplotlib_omission._marks import BarSlashClip, OmissionPatch


@pytest.fixture(autouse=True)
def close_figures():
    """各テストの終了後に図を閉じる。"""
    yield
    plt.close("all")


def test_enable_accepts_omitted_range_after_bar_and_preserves_normal_limits():
    """既存の棒を切り抜き、通常の数値範囲は従来どおり設定する。"""
    matplotlib_omission.enable()
    matplotlib_omission.enable()
    fig, (broken, ordinary) = plt.subplots(1, 2)
    bars = broken.bar([0, 1], [2, 16])
    assert broken.set_ylim(0, 18, step=2, omit=(4, 14), break_style="slash") == (0.0, 18.0)
    assert isinstance(broken.patch, OmissionPatch)
    assert bars.patches[1].get_clip_path()._patch is broken.patch
    assert broken.get_ylim() == (0.0, 18.0)
    np.testing.assert_allclose(broken.get_yticks(), [0, 2, 4, 14, 16, 18])
    assert ordinary.set_ylim(0, 20) == (0.0, 20.0)
    assert ordinary.set_ylim(bottom=1, top=19) == (1.0, 19.0)
    assert ordinary.set_xlim(left=-1, right=2) == (-1.0, 2.0)
    assert not isinstance(ordinary.patch, OmissionPatch)
    fig.canvas.draw()


def test_xlim_and_pyplot_compatibility():
    """横棒と pyplot の範囲設定に省略区間と目盛り間隔を使用できる。"""
    matplotlib_omission.enable()
    fig, ax = plt.subplots()
    ax.barh([0], [16])
    plt.sca(ax)
    assert plt.xlim(left=0, right=18, step=2, omit=(4, 14)) == (0.0, 18.0)
    assert ax.patch.orientation == "horizontal"
    np.testing.assert_allclose(ax.get_xticks(), [0, 2, 4, 14, 16, 18])
    assert plt.ylim() == ax.get_ylim()
    fig.canvas.draw()


def test_existing_chart_api_after_enable():
    """拡張後も既存のグラフ作成 API と通常の範囲設定を使用できる。"""
    matplotlib_omission.enable()
    chart = matplotlib_omission.broken_bar([0], [16], y_ranges=[(0, 4), (14, 18)])
    assert chart.ax.get_ylim() == (0.0, 18.0)
    fig, ax = plt.subplots()
    assert ax.set_ylim(np.array(0), 20) == (0.0, 20.0)
    chart.figure.canvas.draw()


def test_invalid_ranges_and_repeated_configuration():
    """不正な区間や同じ軸への再設定を拒否する。"""
    matplotlib_omission.enable()
    fig, ax = plt.subplots()
    with pytest.raises(ValueError):
        ax.set_ylim(0, 18, omit=(0, 4))
    assert not isinstance(ax.patch, OmissionPatch)
    for omitted in ((4, 18), (14, 4), [(4, 8), (7, 14)]):
        with pytest.raises(ValueError):
            ax.set_ylim(0, 18, omit=omitted)
    for step in (0, -1, float("inf"), 0.0001):
        with pytest.raises(ValueError):
            ax.set_ylim(0, 18, step=step, omit=(4, 14))
    with pytest.raises(ValueError):
        ax.set_ylim(0, omit=(4, 14))
    with pytest.raises(TypeError):
        ax.set_ylim(0, 18, omit=(4, 14), auto=True)
    with pytest.raises(TypeError):
        ax.set_ylim(0, 18, omit=(4, 14), unknown=1)
    ax.set_ylim(0, 18, omit=np.array([[4, 7], [9, 14]]))
    assert ax.patch.bands and len(ax.patch.bands) == 3
    with pytest.raises(ValueError):
        ax.set_xlim(0, 18, omit=(4, 14))
    fig.canvas.draw()


def test_step_without_omission_uses_regular_axes():
    """省略なしの目盛り間隔は通常の軸にも設定できる。"""
    matplotlib_omission.enable()
    fig, ax = plt.subplots()
    assert ax.set_ylim(0, 5, step=1) == (0.0, 5.0)
    np.testing.assert_allclose(ax.get_yticks(), [0, 1, 2, 3, 4, 5])
    assert not isinstance(ax.patch, OmissionPatch)
    fig.canvas.draw()


@pytest.mark.parametrize("orientation", ["horizontal", "vertical"])
@pytest.mark.parametrize("show_labels", [False, True])
def test_break_boundary_tick_labels_can_be_toggled(orientation, show_labels):
    """省略境界の数値だけを切り替え、目盛り位置と他のラベルを保つ。"""
    matplotlib_omission.enable()
    fig, ax = plt.subplots()
    if orientation == "horizontal":
        ax.barh([0], [16])
        ax.set_xlim(0, 18, step=2, omit=(4, 14), show_break_tick_labels=show_labels)
        axis = ax.xaxis
    else:
        ax.bar([0], [16])
        ax.set_ylim(0, 18, step=2, omit=(4, 14), show_break_tick_labels=show_labels)
        axis = ax.yaxis
    fig.canvas.draw()
    np.testing.assert_allclose(axis.get_ticklocs(), [0, 2, 4, 14, 16, 18])
    labels = [axis.get_major_formatter()(value) for value in axis.get_ticklocs()]
    assert bool(labels[2]) is show_labels
    assert bool(labels[3]) is show_labels
    assert all(labels[i] for i in (0, 1, 4, 5))


def test_enabled_bar_slash_clips_bars_added_before_and_after_limits():
    """統合形式でも設定前後に描いた棒の塗りを斜めに切り抜く。"""
    matplotlib_omission.enable()
    fig, ax = plt.subplots()
    first = ax.bar([0], [16], color="red")
    ax.set_ylim(0, 18, omit=(4, 14), break_style="bar_slash")
    second = ax.bar([1], [17], color="blue")
    fig.canvas.draw()
    for bar in (*first.patches, *second.patches):
        assert isinstance(bar.get_clip_path()._patch, BarSlashClip)
