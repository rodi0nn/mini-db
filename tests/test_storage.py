"""
tests/test_storage.py

Tests for both layers of the storage stack:
  - Pager  (mini_db/pager.py)   -> raw page-level file I/O
  - Table  (mini_db/storage.py) -> row (de)serialization + heap file logic

Each test uses pytest's built-in `tmp_path` fixture, which hands us a
fresh temporary directory unique to that test run. This keeps tests
fully isolated from each other and from any real .db file — no
manual cleanup needed, and no risk of one test's leftover data
breaking another test.
"""

import os
from mini_db.pager import Pager, PAGE_SIZE
from mini_db.storage import Table


# --- Pager tests ---------------------------------------------------------
# These test the lowest layer only: raw bytes in, raw bytes out.
# No knowledge of rows or tables should be needed to understand these.

def test_write_and_read_single_page(tmp_path):
    db_file = tmp_path / "test.db"

    pager = Pager(str(db_file))
    data = b"hello database" + b"\x00" * (PAGE_SIZE - len(b"hello database"))
    pager.write_page(0, data)
    pager.close()

    # Reopen fresh, as if it's a brand new process, to prove the
    # data actually persisted to disk rather than just living in memory.
    pager2 = Pager(str(db_file))
    result = pager2.read_page(0)
    pager2.close()

    assert result == data


def test_write_and_read_multiple_pages(tmp_path):
    db_file = tmp_path / "test.db"
    pager = Pager(str(db_file))

    page0 = b"A" * PAGE_SIZE
    page1 = b"B" * PAGE_SIZE
    pager.write_page(0, page0)
    pager.write_page(1, page1)
    pager.close()

    pager2 = Pager(str(db_file))
    assert pager2.read_page(0) == page0
    assert pager2.read_page(1) == page1
    pager2.close()


def test_file_grows_on_disk(tmp_path):
    db_file = tmp_path / "test.db"
    pager = Pager(str(db_file))
    pager.write_page(0, b"X" * PAGE_SIZE)
    pager.write_page(2, b"Y" * PAGE_SIZE)  # skip page 1 on purpose
    pager.close()

    # File should be at least 3 pages on disk, gap included
    assert os.path.getsize(db_file) == 3 * PAGE_SIZE


# --- Table tests -----------------------------------------------------
# These test the layer built on top of Pager: turning Python
# (id, name) rows into bytes, and packing/reading them page by page.

def test_insert_and_scan_single_row(tmp_path):
    db_file = tmp_path / "test.db"

    table = Table(str(db_file))
    table.insert_row(1, "alice")
    table.close()

    # Reopen fresh to prove the row actually persisted, not just
    # held in memory by the same Table instance.
    table2 = Table(str(db_file))
    rows = table2.scan_all_rows()
    table2.close()

    assert rows == [(1, "alice")]


def test_insert_and_scan_multiple_rows_same_page(tmp_path):
    db_file = tmp_path / "test.db"

    table = Table(str(db_file))
    table.insert_row(1, "alice")
    table.insert_row(2, "bob")
    table.insert_row(3, "carol")
    table.close()

    table2 = Table(str(db_file))
    rows = table2.scan_all_rows()
    table2.close()

    # Order should match insertion order, since this is a heap file
    # (append-only, no sorting/indexing yet).
    assert rows == [(1, "alice"), (2, "bob"), (3, "carol")]


def test_insert_enough_rows_to_span_multiple_pages(tmp_path):
    """
    ROWS_PER_PAGE rows fit in a single page. Inserting more than that
    forces Table to allocate a second page — this test exists
    specifically to catch bugs in that page-boundary logic, which
    won't show up in small tests like the ones above.
    """
    from mini_db.storage import ROWS_PER_PAGE

    db_file = tmp_path / "test.db"
    total_rows = ROWS_PER_PAGE + 25  # guarantees at least 2 pages used

    table = Table(str(db_file))
    for i in range(1, total_rows + 1):
        table.insert_row(i, f"user{i}")
    table.close()

    table2 = Table(str(db_file))
    rows = table2.scan_all_rows()
    table2.close()

    assert len(rows) == total_rows
    assert rows[0] == (1, "user1")
    assert rows[-1] == (total_rows, f"user{total_rows}")

    # File on disk should span at least 2 pages, proving the second
    # page was actually allocated rather than data being lost/overwritten.
    assert os.path.getsize(db_file) >= 2 * PAGE_SIZE


def test_name_longer_than_max_len_is_truncated(tmp_path):

    """
    NAME_MAX_LEN is 32 bytes. serialize_row silently truncates
    anything longer rather than raising — this test documents that
    behavior explicitly so it's a known, intentional limitation
    rather than a silent surprise later.
    """
    db_file = tmp_path / "test.db"
    long_name = "x" * 50  # deliberately longer than NAME_MAX_LEN

    table = Table(str(db_file))
    table.insert_row(1, long_name)
    table.close()

    table2 = Table(str(db_file))
    rows = table2.scan_all_rows()
    table2.close()

    stored_id, stored_name = rows[0]
    assert stored_id == 1
    assert stored_name == "x" * 32  # truncated to NAME_MAX_LEN

def test_delete_row_removes_it_from_scan(tmp_path):
    db_file = tmp_path / "test.db"
    
    table = Table(str(db_file))
    table.insert_row(1, "alice")
    table.insert_row(2, "bob")
    table.insert_row(3, "carol")
    table.insert_row(4, "dave")
    table.insert_row(5, "eve")

    deleted = table.delete_row(3)  # delete carol
    table.close()

    table2 = Table(str(db_file))
    rows = table2.scan_all_rows()
    table2.close()

    assert deleted is True
    assert rows == [(1, "alice"), (2, "bob"), (4, "dave"), (5, "eve")]  # carol is gone

def test_delete_nonexistent_row_returns_false(tmp_path):
    db_file = tmp_path / "test.db"

    table = Table(str(db_file))
    table.insert_row(1, "alice")
    deleted = table.delete_row(999)  # non-existent id
    table.close()

    table2 = Table(str(db_file))
    rows = table2.scan_all_rows()
    table2.close()

    assert deleted is False
    assert rows == [(1, "alice")]  # original row still present

def test_delete_then_insert_reuses_freed_space(tmp_path):
    """
    After deleting a row, row_count in that page drops, so a
    subsequent insert should land back in that page rather than
    always allocating a brand new one - proving delete actually
    frees usable space instead of just hiding a row.
    """
    db_file = tmp_path / "test.db"

    table = Table(str(db_file))
    table.insert_row(1, "alice")
    table.insert_row(2, "bob")
    table.delete_row(1)  # delete alice
    table.insert_row(3, "carol")  # should reuse alice's space
    table.close()

    table2 = Table(str(db_file))
    rows = table2.scan_all_rows()
    table2.close()

    assert rows == [(2, "bob"), (3, "carol")]  # alice is gone, carol is present    

