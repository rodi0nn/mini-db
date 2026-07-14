"""
pager.py — Lowest layer of the storage engine.

Responsible ONLY for moving fixed-size "pages" of raw bytes between
memory and a single file on disk. This module has zero knowledge of
rows, columns, tables, or B-trees — it just knows how to address a
file by page number instead of by byte offset.

Why a separate module: every real database (SQLite's pager.c,
Postgres's bufmgr.c) isolates this concern for the same reason —
disk I/O is a fundamentally different responsibility from
interpreting what the bytes mean. Keeping it isolated means storage.py
can be tested without ever thinking about seek/flush/fsync, and this
file can be tested without ever thinking about what a "row" is.

Libraries used:
- `os`: only for filesystem checks (does the file exist yet) and
  measuring file size in bytes. No third-party dependencies —
  intentional, since the point of this project is to avoid anything
  that hides how the disk actually works.
"""

import os

# 4096 bytes is the classic database page size (SQLite's default,
# and a common OS filesystem block size). Picking a size that lines
# up with how the OS/disk already moves data in blocks is why real
# DBs use this number instead of something arbitrary.
PAGE_SIZE = 4096


class Pager:
    """
    Wraps a single database file and exposes it as a numbered
    sequence of fixed-size pages instead of a raw byte stream.

    Page 0 = bytes [0, 4096)
    Page 1 = bytes [4096, 8192)
    Page N = bytes [N * PAGE_SIZE, (N+1) * PAGE_SIZE)

    This numbering is what lets every other layer (storage, B-tree)
    refer to "page 7" as a stable address, regardless of how the
    underlying file grows over time.
    """

    def __init__(self, filepath: str):
        self.filepath = filepath

        # 'r+b' (read/write binary) requires the file to already
        # exist, so we create an empty file first if this is a
        # brand new database.
        if not os.path.exists(filepath):
            open(filepath, "wb").close()

        self.file = open(filepath, "r+b")

    def read_page(self, page_num: int) -> bytes:
        """
        Return exactly PAGE_SIZE bytes for the given page number.

        If the page has never been written (reading past the current
        end of file), we return a zero-filled page rather than
        raising an error. This mirrors how real storage engines treat
        "never written" as a valid, well-defined state (all zero
        bytes) rather than a special case callers need to guard against.
        """
        offset = page_num * PAGE_SIZE
        self.file.seek(offset)
        data = self.file.read(PAGE_SIZE)

        if len(data) < PAGE_SIZE:
            data += b"\x00" * (PAGE_SIZE - len(data))

        return data

    def write_page(self, page_num: int, data: bytes):
        """
        Write exactly PAGE_SIZE bytes at the given page number.

        Two things this handles that are easy to get wrong:

        1. Enforcing the page is EXACTLY PAGE_SIZE bytes. Silently
           accepting a short write would corrupt every page number
           after it (they'd no longer land at multiples of PAGE_SIZE).

        2. Zero-filling any gap if we write a page number beyond the
           current end of file (e.g. file has 1 page, we write page 3
           directly). Without this, page 1 and 2 would be missing
           entirely and reads would return misaligned data.
        """
        if len(data) != PAGE_SIZE:
            raise ValueError(
                f"Page data must be exactly {PAGE_SIZE} bytes, got {len(data)}"
            )

        offset = page_num * PAGE_SIZE
        current_size = os.path.getsize(self.filepath)

        if offset > current_size:
            self.file.seek(current_size)
            self.file.write(b"\x00" * (offset - current_size))

        self.file.seek(offset)
        self.file.write(data)
        self.file.flush()        # push from Python's buffer to the OS
        os.fsync(self.file.fileno())  # push from OS cache to physical disk

    def close(self):
        self.file.close()