"""
storage.py — Row format and page layout, built on top of pager.py.

This is where raw bytes start meaning something. pager.py knows how
to read/write a page; storage.py knows how to pack a Python row
(id, name) into bytes, fit several rows into one page, and read them
back out again.

Layers so far:
    pager.py    -> "give me page N" / "write these bytes to page N"
    storage.py  -> "insert this row" / "give me every row" (THIS FILE)

Libraries used:
- `struct`: converts between Python values (int, str) and fixed-size
  binary layouts, matching how C structs pack data — this is what
  lets us write actual bytes to disk instead of using something like
  `pickle` or `json`, which would hide the byte-level format we're
  trying to learn.
"""

import struct
from mini_db.pager import Pager, PAGE_SIZE

# --- Row format -------------------------------------------------------
#
# Every row is currently hardcoded as: (id: int, name: str)
# This will become dynamic once the SQL front end can define schemas
# (Phase 3), but a fixed schema is deliberate for now — it keeps
# Phase 1/2 focused on storage mechanics rather than schema handling.
#
# Struct format string "I32s":
#   I    -> unsigned int, 4 bytes            (id)
#   32s  -> fixed 32-byte string field       (name, padded/truncated)
NAME_MAX_LEN = 32
ROW_FORMAT = f"I{NAME_MAX_LEN}s"
ROW_SIZE = struct.calcsize(ROW_FORMAT)  # 4 + 32 = 36 bytes per row

# --- Page layout --------------------------------------------------------
#
# A page isn't just "rows back to back" — we need to know how many
# rows are actually stored in a page (the rest is unused/zero space).
# So each page starts with a small header:
#
#   [ row_count: 4 bytes ][ row 0 ][ row 1 ] ... [ unused space ]
#
# "H" would work for row_count (2 bytes, up to 65535), but we use "I"
# (4 bytes) to match the row id field and keep struct formats simple.
PAGE_HEADER_FORMAT = "I"
PAGE_HEADER_SIZE = struct.calcsize(PAGE_HEADER_FORMAT)

# How many whole rows actually fit in a page once the header is
# accounted for. Any leftover bytes in the page just sit unused —
# we never split a row across two pages.
ROWS_PER_PAGE = (PAGE_SIZE - PAGE_HEADER_SIZE) // ROW_SIZE


def serialize_row(row_id: int, name: str) -> bytes:
    """
    Pack a Python row into its fixed 36-byte binary form.

    name is encoded to UTF-8 and padded/truncated to exactly
    NAME_MAX_LEN bytes, since struct's 's' format requires a fixed
    length — it will neither raise on a too-long name nor
    auto-truncate for us, so we do it explicitly.
    """
    name_bytes = name.encode("utf-8")[:NAME_MAX_LEN]
    name_bytes = name_bytes.ljust(NAME_MAX_LEN, b"\x00")  # zero-pad
    return struct.pack(ROW_FORMAT, row_id, name_bytes)


def deserialize_row(data: bytes) -> tuple[int, str]:
    """
    Unpack 36 bytes back into (id, name).

    rstrip(b"\\x00") strips the zero-padding added during
    serialization, then decode() turns the raw bytes back into a
    Python string.
    """
    row_id, name_bytes = struct.unpack(ROW_FORMAT, data)
    name = name_bytes.rstrip(b"\x00").decode("utf-8")
    return row_id, name


class Table:
    """
    Presents a simple insert/scan interface over a heap file —
    "heap" meaning rows are just appended in whatever order they
    arrive, with no ordering or indexing yet. This is intentionally
    the dumbest possible storage strategy: linear scan, no B-tree.
    It exists so Phase 2 (execution engine) has something real to
    call, and so we have a working baseline to compare the B-tree
    against later for speed.
    """

    def __init__(self, filepath: str):
        self.pager = Pager(filepath)

    def insert_row(self, row_id: int, name: str):
        """
        Append a row to the last page that still has room, or start
        a new page if the current last page is full.
        """
        page_num = self._find_page_with_space()
        page = bytearray(self.pager.read_page(page_num))

        row_count = struct.unpack_from(PAGE_HEADER_FORMAT, page, 0)[0]
        row_offset = PAGE_HEADER_SIZE + (row_count * ROW_SIZE)

        row_bytes = serialize_row(row_id, name)
        page[row_offset:row_offset + ROW_SIZE] = row_bytes

        struct.pack_into(PAGE_HEADER_FORMAT, page, 0, row_count + 1)

        self.pager.write_page(page_num, bytes(page))

    def delete_row(self, row_id: int) -> bool:
        """
        Remove the row with the given id from wherever it lives.

        Because this is a heap file with a row_count header per page,
        we can't just zero out the deleted row's bytes and leave a
        hole in the middle — scan_all_rows() trusts that rows
        [0, row_count) are all valid, contiguous entries. So instead
        we shift every row after the deleted one left by one slot,
        then decrement row_count. This keeps the "no gaps" invariant
        that the rest of the class depends on.

        Returns True if a row was found and deleted, False otherwise
        — callers (e.g. the future SQL executor) need to distinguish
        "deleted successfully" from "no such row" to report errors
        correctly.
        """
        page_num = 0

        while self._page_exists_on_disk(page_num):
            page = bytearray(self.pager.read_page(page_num))
            row_count = struct.unpack_from(PAGE_HEADER_FORMAT, page, 0)[0]

            for i in range(row_count):
                offset = PAGE_HEADER_SIZE + (i * ROW_SIZE)
                existing_id, _ = deserialize_row(page[offset:offset + ROW_SIZE])

                if existing_id == row_id:
                    # Shift every row after this one by one slot
                    # overwriting the deleted row in the process.
                    for j in range(i +1, row_count):
                        src_offset = PAGE_HEADER_SIZE + (j * ROW_SIZE)
                        dst_offset = PAGE_HEADER_SIZE + ((j - 1) * ROW_SIZE)
                        page[dst_offset:dst_offset + ROW_SIZE] = page[src_offset:src_offset + ROW_SIZE]

                    struct.pack_into(PAGE_HEADER_FORMAT, page, 0, row_count - 1)
                    self.pager.write_page(page_num, bytes(page))
                    return True  # row deleted successfully
            
            page_num += 1  # move to the next page

        return False  # row_id not found in any page

    def scan_all_rows(self) -> list[tuple[int, str]]:
        """
        Linear scan: walk every page, read every row in it. This is
        the "table scan" you hear about in query planners — the
        slow path that a B-tree index (Phase 4) exists to avoid.
        """
        rows = []
        page_num = 0

        while True:
            page = self.pager.read_page(page_num)
            row_count = struct.unpack_from(PAGE_HEADER_FORMAT, page, 0)[0]

            if row_count == 0 and not self._page_exists_on_disk(page_num):
                break  # ran past the end of the file

            for i in range(row_count):
                offset = PAGE_HEADER_SIZE + (i * ROW_SIZE)
                row_bytes = page[offset:offset + ROW_SIZE]
                rows.append(deserialize_row(row_bytes))

            page_num += 1

        return rows

    def close(self):
        self.pager.close()

    # --- internal helpers -------------------------------------------

    def _find_page_with_space(self) -> int:
        page_num = 0
        while self._page_exists_on_disk(page_num):
            page = self.pager.read_page(page_num)
            row_count = struct.unpack_from(PAGE_HEADER_FORMAT, page, 0)[0]
            if row_count < ROWS_PER_PAGE:
                return page_num
            page_num += 1
        return page_num  # first page number that doesn't exist yet

    def _page_exists_on_disk(self, page_num: int) -> bool:
        import os
        if not os.path.exists(self.pager.filepath):
            return False
        file_size = os.path.getsize(self.pager.filepath)
        return (page_num + 1) * PAGE_SIZE <= file_size
    
