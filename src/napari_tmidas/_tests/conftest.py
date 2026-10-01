import pytest


@pytest.fixture(autouse=True)
def _pip_install_commands(monkeypatch):
    """Build dedicated-env installs as pip commands, whether or not uv is
    installed, so tests asserting on command arguments see one layout.
    Tests of the uv path remove the variable themselves."""
    monkeypatch.setenv("NAPARI_TMIDAS_NO_UV", "1")


@pytest.fixture(autouse=True)
def _no_workflow_log(monkeypatch):
    """Keep tmidas_workflow.jsonl/.md out of the folders other tests list.
    Workflow-log tests delete the variable themselves."""
    monkeypatch.setenv("NAPARI_TMIDAS_NO_WORKFLOW_LOG", "1")
    monkeypatch.delenv("NAPARI_TMIDAS_WORKFLOW_LOG", raising=False)
