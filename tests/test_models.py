import hashlib

import pytest

from shardline import cli
from shardline.models import add_model, default_model_dir, model_path, sha256_of

CONTENT = b"GGUF" + bytes(range(256)) * 40


def make_model(tmp_path, content=CONTENT):
    path = tmp_path / "source.gguf"
    path.write_bytes(content)
    return path


def test_a_model_is_stored_under_its_digest(tmp_path):
    digest = add_model(make_model(tmp_path), tmp_path / "models")

    assert digest == hashlib.sha256(CONTENT).hexdigest()
    stored = tmp_path / "models" / f"{digest}.gguf"
    assert stored.read_bytes() == CONTENT
    assert sha256_of(stored) == digest
    assert [p.name for p in (tmp_path / "models").iterdir()] == [stored.name]


def test_adding_the_same_model_twice_changes_nothing(tmp_path):
    source = make_model(tmp_path)
    models = tmp_path / "models"
    assert add_model(source, models) == add_model(source, models)
    assert len(list(models.iterdir())) == 1


def test_a_file_that_is_not_gguf_is_refused_and_leaves_nothing_behind(tmp_path):
    models = tmp_path / "models"
    with pytest.raises(ValueError, match="not a GGUF"):
        add_model(make_model(tmp_path, b"not a model"), models)
    assert list(models.iterdir()) == []


def test_a_missing_file_is_an_error(tmp_path):
    with pytest.raises(OSError):
        add_model(tmp_path / "absent.gguf", tmp_path / "models")


def test_only_a_digest_names_a_model(tmp_path):
    assert model_path(tmp_path, "a" * 64) == tmp_path / f"{'a' * 64}.gguf"
    for bad in ["", "a" * 63, "a" * 65, "A" * 64, "../" + "a" * 61, "a" * 63 + "/"]:
        with pytest.raises(ValueError):
            model_path(tmp_path, bad)


def test_the_model_directory_can_be_set_by_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("SHARDLINE_MODEL_DIR", str(tmp_path / "from-env"))
    assert default_model_dir() == tmp_path / "from-env"
    monkeypatch.delenv("SHARDLINE_MODEL_DIR")
    assert default_model_dir().parts[-3:] == (".cache", "shardline", "models")


def test_cli_model_add_prints_the_digest(tmp_path, capsys):
    code = cli.main(["model", "add", str(make_model(tmp_path)), "--model-dir", str(tmp_path / "m")])
    assert code == 0
    assert capsys.readouterr().out.strip() == hashlib.sha256(CONTENT).hexdigest()


def test_cli_model_add_reports_a_bad_file(tmp_path, capsys):
    bad = make_model(tmp_path, b"nope")
    assert cli.main(["model", "add", str(bad), "--model-dir", str(tmp_path / "m")]) == 1
    assert "not a GGUF" in capsys.readouterr().err
