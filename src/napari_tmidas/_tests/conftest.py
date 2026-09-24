import pytest


@pytest.fixture(autouse=True)
def _pip_install_commands(monkeypatch):
    """Build dedicated-env installs as pip commands, whether or not uv is
    installed, so tests asserting on command arguments see one layout.
    Tests of the uv path remove the variable themselves."""
    monkeypatch.setenv("NAPARI_TMIDAS_NO_UV", "1")
