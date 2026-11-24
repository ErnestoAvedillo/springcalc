import matplotlib
import matplotlib.pyplot as plt
import pytest
from matplotlib._pylab_helpers import Gcf

from springcalc.lineal import plotting
from springcalc.lineal.plotting import interactive_backend


def test_show_true_hides_and_restores_figures_left_open(monkeypatch):
    # Use Agg as the "interactive" backend so the test never opens a window;
    # what matters is that figures left open before the call are hidden from
    # plt.show() inside the context and restored afterwards.
    monkeypatch.setattr(plotting, "INTERACTIVE_BACKENDS", ["Agg"])
    leaked = plt.figure()
    try:
        with interactive_backend(True):
            assert leaked.number not in Gcf.figs
            inner = plt.figure()
            plt.close(inner)
        assert leaked.number in Gcf.figs
        assert matplotlib.get_backend().lower() == "agg"
    finally:
        plt.close(leaked)


def test_figures_restored_when_no_backend_available(monkeypatch):
    monkeypatch.setattr(plotting, "INTERACTIVE_BACKENDS", ["NoSuchBackend"])
    leaked = plt.figure()
    try:
        with pytest.raises(RuntimeError):
            with interactive_backend(True):
                pass
        assert leaked.number in Gcf.figs
    finally:
        plt.close(leaked)
