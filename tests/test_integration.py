# Copyright (c) 2026 magichan (https://github.com/magichan-lab)

"""Matplotlib の軸範囲 API から省略区間を指定する動作を検証する。"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

import matplotlib_omission
from matplotlib_omission._marks import OmissionPatch


@pytest.fixture(autouse=True)
def close_figures():
    """各テストの終了後に図を閉じる。"""
    yield
    plt.close("all")


def test_enable_accepts_visible_ranges_after_bar_and_preserves_normal_limits():
    """既存の棒を切り抜き、通常の数値範囲は従来どおり設定する。"""
    matplotlib_omission.enable()
    matplotlib_omission.enable()
    fig, (broken, ordinary) = plt.subplots(1, 2)
    bars = broken.bar([0, 1], [2, 16])
    assert broken.set_ylim([(0, 4), (14, 18)], break_style="slash") == (0.0, 18.0)
    assert isinstance(broken.patch, OmissionPatch)
    assert bars.patches[1].get_clip_path()._patch is broken.patch
    assert broken.get_ylim() == (0.0, 18.0)
    assert ordinary.set_ylim(0, 20) == (0.0, 20.0)
    assert ordinary.set_ylim(bottom=1, top=19) == (1.0, 19.0)
    assert ordinary.set_xlim(left=-1, right=2) == (-1.0, 2.0)
    assert not isinstance(ordinary.patch, OmissionPatch)
    fig.canvas.draw()


def test_xlim_and_pyplot_compatibility():
    """横棒と pyplot の範囲設定に同じ区間形式を使用できる。"""
    matplotlib_omission.enable()
    fig, ax = plt.subplots()
    ax.barh([0], [16])
    plt.sca(ax)
    assert plt.xlim(left=[(0, 4), (14, 18)]) == (0.0, 18.0)
    assert ax.patch.orientation == "horizontal"
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
        ax.set_ylim([(0, 4), (3, 8)])
    assert not isinstance(ax.patch, OmissionPatch)
    with pytest.raises(TypeError):
        ax.set_ylim([(0, 4), (14, 18)], top=20)
    with pytest.raises(TypeError):
        ax.set_ylim([(0, 4), (14, 18)], auto=True)
    with pytest.raises(TypeError):
        ax.set_ylim([(0, 4), (14, 18)], unknown=1)
    ax.set_ylim(np.array([[0, 4], [14, 18]]))
    with pytest.raises(ValueError):
        ax.set_xlim([(0, 4), (14, 18)])
    fig.canvas.draw()
