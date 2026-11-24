from springcalc.lineal import plotting


def pytest_configure(config):
    """Keep the test run headless: graph methods called with show=True still
    run their show code path, but against the non-interactive Agg backend, so
    no window opens and nothing blocks waiting for it to be closed.

    Done here, at session start, rather than in a fixture, because some test
    modules (e.g. test_lineal_extension.py) run plotting code at import time,
    during collection, before any fixture is active."""
    plotting.INTERACTIVE_BACKENDS = ["Agg"]
