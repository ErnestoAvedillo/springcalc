"""Shared matplotlib backend handling for the library's plotting methods."""
from contextlib import contextmanager
import matplotlib
from matplotlib._pylab_helpers import Gcf

# Tried in this order; whichever has its GUI bindings installed wins.
INTERACTIVE_BACKENDS = ['TkAgg', 'QtAgg', 'Qt5Agg', 'GTK3Agg', 'GTK4Agg']


@contextmanager
def interactive_backend(show: bool):
    """
    Temporarily switch matplotlib to a real interactive backend so plt.show()
    can open a window, then restore whatever backend was active before.

    The library pins matplotlib to the non-interactive "Agg" backend by
    default (needed for headless web/PDF PNG generation), under which
    plt.show() is a silent no-op. This is only used when show is True;
    otherwise it's a no-op and the current backend is left untouched.
    """
    if not show:
        yield
        return

    # Figures already open were created under the previous (usually Agg)
    # backend. plt.show() walks every open figure, and with a GUI backend it
    # crashes on the first non-GUI figure manager it meets. Hide them from
    # pyplot while the interactive window is shown and put them back after.
    hidden_figures = dict(Gcf.figs)
    Gcf.figs.clear()

    original_backend = matplotlib.get_backend()
    for backend in INTERACTIVE_BACKENDS:
        try:
            matplotlib.use(backend, force=True)
            break
        except Exception:
            continue
    else:
        Gcf.figs.update(hidden_figures)
        raise RuntimeError(
            "show=True needs an interactive matplotlib backend "
            "(Tk/Qt/GTK bindings), none of which are available here."
        )
    try:
        yield
    finally:
        matplotlib.use(original_backend, force=True)
        Gcf.figs.update(hidden_figures)
