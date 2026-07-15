"""
executor.py — Phase 2: naive execution engine.

This is the layer that will eventually sit underneath the SQL parser:
the parser's job (Phase 3) will be to turn "SELECT * FROM users WHERE
id = 3" into a call like executor.select(where_id=3). For now, with
no parser yet, we call these methods directly with plain Python
arguments — the point of this phase is to prove the execution layer
itself works, independent of whether the input came from parsed SQL
or a Python function call.

Layers so far:
    pager.py     -> raw page-level file I/O
    storage.py   -> row (de)serialization + heap file (Table)
    executor.py  -> query-shaped operations on top of Table (THIS FILE)

Why a separate layer instead of just calling Table directly everywhere:
Table only knows about raw (id, name) tuples and heap-file mechanics.
Executor is where query semantics start to exist — e.g. "give me rows
matching this condition" — even though right now that condition logic
is trivial (a single equality check). This is deliberately the
smallest possible thing that deserves to be called a query executor,
so that Phase 3 has an obvious, already-tested target to wire the
parser's AST into.
"""


from mini_db.storage import Table

class Executor:
    """
    Hardcoded-schema query executor: every table has rows shaped
    (id: int, name: str). No CREATE TABLE, no multiple tables, no
    real WHERE clause parsing yet — those come in Phase 3+.
    """

    def __init__(self, db_path:str):
        self.table = Table(db_path)

    def insert(self, row_id:int, name:str):
        """
        Executes what will eventually be INSERT INTO users VALUES (id, name).
        """
        self.table.insert_row(row_id, name)

    def select(self, where_id:int|None=None) -> list[tuple[int, str]]:
        """
        Executes what will eventually be:
            SELECT * FROM users                  (where_id=None)
            SELECT * FROM users WHERE id = <n>    (where_id=n)

        The filtering happens here, in the executor, rather than in
        Table — Table's job is just to hand back every row it has;
        deciding which rows match a condition is query logic, which
        belongs in this layer. This separation is what will let
        Phase 4 swap in a B-tree lookup for the where_id case (an
        O(log n) point lookup) without SELECT's calling code changing
        at all — only what happens inside this method changes.
        """
        rows = self.table.scan_all_rows()

        if where_id is None:
            return rows
        
        return [row for row in rows if row[0] == where_id]

    def delete(self, row_id:int) -> bool:
        """
        Executes what will eventually be DELETE FROM users WHERE id = <n>.

        Returns True/False (forwarded from Table.delete_row) so
        callers can distinguish "deleted" from "no such row" —
        eventually this becomes what the SQL front end reports back
        to the user (e.g. "DELETE 1" vs "DELETE 0" / an error).
        """
        return self.table.delete_row(row_id)

    def close(self) -> None:
        self.table.close()
