"""Dependency-light source-exact tar extraction checks for torchvision #9693.

This test deliberately compiles the actual function ASTs from the checked-out
torchvision/datasets/utils.py, not a hand-transcribed copy or an installed
binary wheel. Other torchvision package imports are excluded to keep Python
3.11 (including the legacy filter path) testable in isolation.
"""
from __future__ import annotations

import ast
import bz2
import contextlib
import copy
import gzip
import io
import lzma
import os
import pathlib
import tarfile
import tempfile
import unittest
import zipfile

SOURCE = pathlib.Path(__file__).resolve().parents[1] / "torchvision/datasets/utils.py"
FUNCTIONS = {
    "_extract_tar", "_filter_tar_member", "_validate_tar_member",
    "_sanitize_legacy_tar_member", "_is_within_directory",
    "_extract_zip", "_detect_file_type", "_decompress", "extract_archive",
}
CONSTANTS = {
    "_ZIP_COMPRESSION_MAP", "_ARCHIVE_EXTRACTORS",
    "_COMPRESSED_FILE_OPENERS", "_FILE_TYPE_ALIASES",
}


def load_source_exact():
    source = SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(SOURCE))
    nodes = [
        n for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in FUNCTIONS
        or isinstance(n, ast.AnnAssign)
        and isinstance(n.target, ast.Name) and n.target.id in CONSTANTS
    ]
    found = {
        n.name if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) else n.target.id
        for n in nodes
    }
    assert found == FUNCTIONS | CONSTANTS, (FUNCTIONS | CONSTANTS) - found
    mod = ast.Module(body=[
        ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
        *nodes,
    ], type_ignores=[])
    ast.fix_missing_locations(mod)
    scope = {
        "__name__": "torchvision_source_exact_tar",
        "os": os, "pathlib": pathlib, "tarfile": tarfile, "copy": copy,
        "bz2": bz2, "gzip": gzip, "lzma": lzma, "zipfile": zipfile,
    }
    exec(compile(mod, str(SOURCE), "exec"), scope)
    return scope


EXACT = load_source_exact()


@contextlib.contextmanager
def filter_mode(legacy: bool):
    # Simulate Python 3.11 versions where tarfile.data_filter is absent.
    original = getattr(tarfile, "data_filter", None)
    if legacy and original is not None:
        delattr(tarfile, "data_filter")
    try:
        yield
    finally:
        if legacy and original is not None:
            tarfile.data_filter = original


class SourceExactTarTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = pathlib.Path(self.tmp.name)
        self.dest = self.root / "extracted"
        self.dest.mkdir()

    def archive(self, suffix, members):
        mode = {".tar": "w", ".tar.gz": "w:gz", ".tgz": "w:gz",
                ".tar.xz": "w:xz"}[suffix]
        path = self.root / ("sample" + suffix)
        with tarfile.open(path, mode) as tar:
            for member, payload in members:
                if payload is not None:
                    member.size = len(payload)
                    tar.addfile(member, io.BytesIO(payload))
                else:
                    tar.addfile(member)
        return path

    def regular(self, name, payload=b"data"):
        return (tarfile.TarInfo(name), payload)

    def extract(self, archive, legacy):
        with filter_mode(legacy):
            return EXACT["extract_archive"](archive, self.dest)

    def test_compressed_and_uncompressed_parent_traversal(self):
        for suffix in (".tar", ".tar.gz", ".tgz", ".tar.xz"):
            for legacy in (True, False):
                with self.subTest(suffix=suffix, legacy=legacy):
                    if not legacy and not hasattr(tarfile, "data_filter"):
                        continue
                    path = self.archive(suffix, [self.regular("../outside.txt", b"escape")])
                    with self.assertRaises(tarfile.TarError):
                        self.extract(path, legacy)
                    self.assertFalse((self.root / "outside.txt").exists())

    def test_absolute_archive_path(self):
        outside = self.root / "absolute.txt"
        for legacy in (True, False):
            with self.subTest(legacy=legacy):
                if not legacy and not hasattr(tarfile, "data_filter"):
                    continue
                path = self.archive(".tar", [self.regular(str(outside), b"escape")])
                with self.assertRaises(tarfile.TarError):
                    self.extract(path, legacy)
                self.assertFalse(outside.exists())

    def test_outside_link_targets(self):
        for legacy in (True, False):
            for kind in (tarfile.SYMTYPE, tarfile.LNKTYPE):
                with self.subTest(legacy=legacy, kind=kind):
                    if not legacy and not hasattr(tarfile, "data_filter"):
                        continue
                    item = tarfile.TarInfo("escape")
                    item.type = kind
                    item.linkname = "../outside"
                    path = self.archive(".tar", [(item, None)])
                    with self.assertRaises(tarfile.TarError):
                        self.extract(path, legacy)
                    self.assertFalse((self.root / "outside").exists())

    @unittest.skipIf(os.name == "nt", "requires destination symlink creation")
    def test_preexisting_destination_symlink_escape(self):
        outside = self.root / "outside"
        outside.mkdir()
        (self.dest / "redirect").symlink_to(outside, target_is_directory=True)
        path = self.archive(".tar.gz", [self.regular("redirect/payload", b"escape")])
        for legacy in (True, False):
            with self.subTest(legacy=legacy):
                if not legacy and not hasattr(tarfile, "data_filter"):
                    continue
                with self.assertRaises(tarfile.TarError):
                    self.extract(path, legacy)
                self.assertFalse((outside / "payload").exists())

    def test_internal_hardlink_is_preserved(self):
        first = self.regular("target.txt", b"safe")
        link = tarfile.TarInfo("nested/hardlink.txt")
        link.type = tarfile.LNKTYPE
        link.linkname = "target.txt"
        path = self.archive(".tar", [first, (link, None)])
        for legacy in (True, False):
            with self.subTest(legacy=legacy):
                if not legacy and not hasattr(tarfile, "data_filter"):
                    continue
                self.extract(path, legacy)
                self.assertEqual((self.dest / "nested/hardlink.txt").read_bytes(), b"safe")

    @unittest.skipIf(os.name == "nt", "symlink privileges differ")
    def test_internal_symlink_is_preserved(self):
        item = self.regular("nested/target.txt", b"safe")
        link = tarfile.TarInfo("safe-link")
        link.type = tarfile.SYMTYPE
        link.linkname = "nested/target.txt"
        path = self.archive(".tar", [item, (link, None)])
        for legacy in (True, False):
            with self.subTest(legacy=legacy):
                if not legacy and not hasattr(tarfile, "data_filter"):
                    continue
                self.extract(path, legacy)
                self.assertEqual((self.dest / "safe-link").read_bytes(), b"safe")

    def test_special_fifo_rejected(self):
        entry = tarfile.TarInfo("fifo")
        entry.type = tarfile.FIFOTYPE
        path = self.archive(".tar", [(entry, None)])
        for legacy in (True, False):
            with self.subTest(legacy=legacy):
                if not legacy and not hasattr(tarfile, "data_filter"):
                    continue
                with self.assertRaises(tarfile.TarError):
                    self.extract(path, legacy)

    def test_legacy_member_metadata_restrictions(self):
        item = tarfile.TarInfo("normal.txt")
        item.mode = 0o7777
        item.uid = 12345
        item.gid = 6789
        item.uname = "untrusted"
        item.gname = "untrusted"
        safe = EXACT["_sanitize_legacy_tar_member"](item)
        self.assertEqual(safe.mode, 0o755)
        self.assertEqual((safe.uid, safe.gid, safe.uname, safe.gname),
                         (None, None, None, None))
        self.assertEqual(item.mode, 0o7777)
        path = self.archive(".tar", [(item, b"safe")])
        self.extract(path, legacy=True)
        self.assertEqual((self.dest / "normal.txt").read_bytes(), b"safe")


if __name__ == "__main__":
    unittest.main(verbosity=2)
