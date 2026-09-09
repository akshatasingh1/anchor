from pathlib import Path

import pytest

from extract import extract_file, extract_repo

SAMPLE = '''\
import os

TOP_LEVEL = 1


def free_function(x):
    return x + 1


class Widget:
    """A widget."""

    def __init__(self, size):
        self.size = size

    def area(self):
        def helper():
            return 2

        return self.size * helper()


async def fetch(url):
    return url
'''


@pytest.fixture
def sample_file(tmp_path: Path) -> Path:
    f = tmp_path / "sample.py"
    f.write_text(SAMPLE)
    return f


def _by_name(symbols):
    return {s.name: s for s in symbols}


def test_extracts_all_symbol_kinds(sample_file):
    syms = _by_name(extract_file(sample_file, sample_file.parent))

    assert syms["free_function"].kind == "function"
    assert syms["Widget"].kind == "class"
    assert syms["Widget.__init__"].kind == "method"
    assert syms["Widget.area"].kind == "method"
    assert syms["fetch"].kind == "function"


def test_methods_get_qualified_names(sample_file):
    names = {s.name for s in extract_file(sample_file, sample_file.parent)}
    assert "Widget.area" in names
    assert "area" not in names  # bare name should not leak


def test_nested_function_is_not_qualified_to_class(sample_file):
    # helper() is defined inside a method; class context must reset for it
    syms = _by_name(extract_file(sample_file, sample_file.parent))
    assert "helper" in syms
    assert syms["helper"].kind == "function"


def test_line_numbers_and_location(sample_file):
    syms = _by_name(extract_file(sample_file, sample_file.parent))
    ff = syms["free_function"]
    assert ff.start_line == 6
    assert ff.location() == "sample.py:6"
    assert ff.source.startswith("def free_function")


def test_extract_repo_skips_ignored_dirs(tmp_path: Path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "a.py").write_text("def keep(): pass\n")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "hook.py").write_text("def drop(): pass\n")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "dep.py").write_text("def also_drop(): pass\n")

    names = {s.name for s in extract_repo(tmp_path)}
    assert names == {"keep"}


def test_extract_repo_tolerates_unparseable_file(tmp_path: Path, capsys):
    (tmp_path / "ok.py").write_text("def good(): pass\n")
    (tmp_path / "bad.py").write_bytes(b"\xff\xfe not utf-8 \x00 def (")

    names = {s.name for s in extract_repo(tmp_path)}
    assert "good" in names
