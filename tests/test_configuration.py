from pathlib import Path

from edugraph_classify.configuration import load_local_environment


def test_environment_file_loads_without_overriding_process_values(
    tmp_path: Path, monkeypatch
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("EXISTING=from-file\nNEW_VALUE=loaded\n", encoding="utf-8")
    monkeypatch.setenv("EXISTING", "from-process")
    monkeypatch.delenv("NEW_VALUE", raising=False)

    assert load_local_environment(env_file) is True
    assert __import__("os").environ["EXISTING"] == "from-process"
    assert __import__("os").environ["NEW_VALUE"] == "loaded"


def test_missing_environment_file_is_a_noop(tmp_path: Path) -> None:
    assert load_local_environment(tmp_path / "missing.env") is False
