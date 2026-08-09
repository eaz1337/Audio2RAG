import logging

from extract.discover import discover_audio_files

SUPPORTED = ["mp3", "wav", "m4a"]


def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"data")


def test_finds_supported_extensions_only(tmp_path):
    _touch(tmp_path / "a.wav")
    _touch(tmp_path / "b.mp3")
    _touch(tmp_path / "notes.txt")

    found = discover_audio_files(tmp_path, SUPPORTED)

    assert sorted(p.name for p in found) == ["a.wav", "b.mp3"]


def test_extension_match_is_case_insensitive(tmp_path):
    _touch(tmp_path / "a.WAV")

    found = discover_audio_files(tmp_path, SUPPORTED)

    assert [p.name for p in found] == ["a.WAV"]


def test_non_recursive_ignores_subdirectories(tmp_path):
    _touch(tmp_path / "top.wav")
    _touch(tmp_path / "sub" / "nested.wav")

    found = discover_audio_files(tmp_path, SUPPORTED, recursive=False)

    assert [p.name for p in found] == ["top.wav"]


def test_recursive_finds_nested_files(tmp_path):
    _touch(tmp_path / "top.wav")
    _touch(tmp_path / "sub" / "nested.wav")
    _touch(tmp_path / "sub" / "deeper" / "deepest.mp3")

    found = discover_audio_files(tmp_path, SUPPORTED, recursive=True)

    assert sorted(p.name for p in found) == ["deepest.mp3", "nested.wav", "top.wav"]


def test_unsupported_file_is_skipped_with_warning(tmp_path, caplog):
    _touch(tmp_path / "notes.txt")

    with caplog.at_level(logging.WARNING):
        found = discover_audio_files(tmp_path, SUPPORTED)

    assert found == []
    assert "notes.txt" in caplog.text
