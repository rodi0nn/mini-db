"""
tests/test_parser.py

Tests for mini_db/parser.py. Each test tokenizes real SQL text first,
then parses it — testing the full "text in, AST out" pipeline rather
than hand-building token lists, since that's how the parser will
actually be used once wired into the executor.
"""

import pytest
from mini_db.tokenizer import tokenize
from mini_db.parser import (
    parse,
    SelectStatement,
    InsertStatement,
    DeleteStatement,
    ParserError,
)


# --- SELECT ------------------------------------------------------------

def test_parse_select_all():
    ast = parse(tokenize("SELECT * FROM users;"))
    assert ast == SelectStatement(table="users")


def test_parse_select_with_where_integer():
    ast = parse(tokenize("SELECT * FROM users WHERE id = 3;"))
    assert ast == SelectStatement(table="users", where_column="id", where_value=3)


def test_parse_select_without_semicolon():
    # SQL statements should parse fine with or without a trailing ;
    ast = parse(tokenize("SELECT * FROM users"))
    assert ast == SelectStatement(table="users")


# --- INSERT ------------------------------------------------------------

def test_parse_insert():
    ast = parse(tokenize("INSERT INTO users VALUES (1, 'alice');"))
    assert ast == InsertStatement(table="users", values=(1, "alice"))


def test_parse_insert_without_semicolon():
    ast = parse(tokenize("INSERT INTO users VALUES (2, 'bob')"))
    assert ast == InsertStatement(table="users", values=(2, "bob"))


# --- DELETE ------------------------------------------------------------

def test_parse_delete_with_where():
    ast = parse(tokenize("DELETE FROM users WHERE id = 3;"))
    assert ast == DeleteStatement(table="users", where_column="id", where_value=3)


def test_parse_delete_without_where():
    ast = parse(tokenize("DELETE FROM users;"))
    assert ast == DeleteStatement(table="users")


# --- error cases ----------------------------------------------------------

def test_select_missing_from_raises_parser_error():
    with pytest.raises(ParserError):
        parse(tokenize("SELECT * users;"))


def test_select_missing_table_name_raises_parser_error():
    with pytest.raises(ParserError):
        parse(tokenize("SELECT * FROM WHERE id = 3;"))


def test_insert_missing_values_keyword_raises_parser_error():
    with pytest.raises(ParserError):
        parse(tokenize("INSERT INTO users (1, 'alice');"))


def test_insert_with_wrong_value_order_raises_parser_error():
    # Schema is hardcoded as (id: int, name: string) - int must come first
    with pytest.raises(ParserError):
        parse(tokenize("INSERT INTO users VALUES ('alice', 1);"))


def test_delete_missing_from_raises_parser_error():
    with pytest.raises(ParserError):
        parse(tokenize("DELETE users WHERE id = 3;"))


def test_statement_not_starting_with_known_keyword_raises_parser_error():
    with pytest.raises(ParserError):
        parse(tokenize("UPDATE users SET id = 3;"))


def test_trailing_garbage_after_statement_raises_parser_error():
    # Extra tokens after a complete, valid statement should be
    # rejected, not silently ignored.
    with pytest.raises(ParserError):
        parse(tokenize("SELECT * FROM users EXTRA STUFF;"))


def test_where_clause_missing_equals_raises_parser_error():
    with pytest.raises(ParserError):
        parse(tokenize("SELECT * FROM users WHERE id 3;"))