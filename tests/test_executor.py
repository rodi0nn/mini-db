"""
tests/test_executor.py

Tests for the Executor layer (mini_db/executor.py). These deliberately
call insert/select/delete with plain Python arguments, standing in
for what the SQL parser will eventually produce — the point of Phase
2 is proving this layer works correctly before Phase 3 wires a parser
into it.
"""

from mini_db.executor import Executor


def test_insert_then_select_all_returns_the_row(tmp_path):
    db_file = tmp_path / "test.db"

    ex = Executor(str(db_file))
    ex.insert(1, "alice")
    rows = ex.select()
    ex.close()

    assert rows == [(1, "alice")]


def test_select_all_on_empty_table_returns_empty_list(tmp_path):
    db_file = tmp_path / "test.db"

    ex = Executor(str(db_file))
    rows = ex.select()
    ex.close()

    assert rows == []


def test_select_with_where_id_returns_only_matching_row(tmp_path):
    db_file = tmp_path / "test.db"

    ex = Executor(str(db_file))
    ex.insert(1, "alice")
    ex.insert(2, "bob")
    ex.insert(3, "carol")

    result = ex.select(where_id=2)
    ex.close()

    assert result == [(2, "bob")]


def test_select_with_where_id_that_does_not_exist_returns_empty_list(tmp_path):
    db_file = tmp_path / "test.db"

    ex = Executor(str(db_file))
    ex.insert(1, "alice")

    result = ex.select(where_id=999)
    ex.close()

    assert result == []


def test_delete_removes_row_from_subsequent_select(tmp_path):
    db_file = tmp_path / "test.db"

    ex = Executor(str(db_file))
    ex.insert(1, "alice")
    ex.insert(2, "bob")

    deleted = ex.delete(1)
    rows = ex.select()
    ex.close()

    assert deleted is True
    assert rows == [(2, "bob")]


def test_delete_nonexistent_row_returns_false_and_changes_nothing(tmp_path):
    db_file = tmp_path / "test.db"

    ex = Executor(str(db_file))
    ex.insert(1, "alice")

    deleted = ex.delete(999)
    rows = ex.select()
    ex.close()

    assert deleted is False
    assert rows == [(1, "alice")]


def test_operations_persist_across_executor_instances(tmp_path):
    """
    End-to-end check that Executor -> Table -> Pager -> disk actually
    round-trips correctly through the whole stack, not just within
    a single Executor instance's lifetime.
    """
    db_file = tmp_path / "test.db"

    ex1 = Executor(str(db_file))
    ex1.insert(1, "alice")
    ex1.insert(2, "bob")
    ex1.delete(1)
    ex1.close()

    ex2 = Executor(str(db_file))
    rows = ex2.select()
    ex2.close()

    assert rows == [(2, "bob")]
    