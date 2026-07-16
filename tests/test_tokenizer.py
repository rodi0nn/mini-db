"""
tests/test_tokenizer.py

Tests for mini_db/tokenizer.py. These check that raw SQL text is
correctly split into the expected token sequence — no grammar
validation happens here (that's the parser's job, tested separately
once it exists), just "did we recognize the pieces correctly."
"""

import pytest
from mini_db.tokenizer import tokenize, Token, TokenType, TokenizerError


def test_tokenize_select_star():
    tokens = tokenize("SELECT * FROM users;")

    assert tokens == [
        Token(TokenType.SELECT),
        Token(TokenType.ASTERISK),
        Token(TokenType.FROM),
        Token(TokenType.IDENTIFIER, "users"),
        Token(TokenType.SEMICOLON),
        Token(TokenType.EOF),
    ]


def test_tokenize_select_with_where():
    tokens = tokenize("SELECT * FROM users WHERE id = 3;")

    assert tokens == [
        Token(TokenType.SELECT),
        Token(TokenType.ASTERISK),
        Token(TokenType.FROM),
        Token(TokenType.IDENTIFIER, "users"),
        Token(TokenType.WHERE),
        Token(TokenType.IDENTIFIER, "id"),
        Token(TokenType.EQUALS),
        Token(TokenType.INTEGER, 3),
        Token(TokenType.SEMICOLON),
        Token(TokenType.EOF),
    ]


def test_tokenize_insert_with_string_value():
    tokens = tokenize("INSERT INTO users VALUES (1, 'alice');")

    assert tokens == [
        Token(TokenType.INSERT),
        Token(TokenType.INTO),
        Token(TokenType.IDENTIFIER, "users"),
        Token(TokenType.VALUES),
        Token(TokenType.LPAREN),
        Token(TokenType.INTEGER, 1),
        Token(TokenType.COMMA),
        Token(TokenType.STRING, "alice"),
        Token(TokenType.RPAREN),
        Token(TokenType.SEMICOLON),
        Token(TokenType.EOF),
    ]


def test_tokenize_delete_with_where():
    tokens = tokenize("DELETE FROM users WHERE id = 3;")

    assert tokens == [
        Token(TokenType.DELETE),
        Token(TokenType.FROM),
        Token(TokenType.IDENTIFIER, "users"),
        Token(TokenType.WHERE),
        Token(TokenType.IDENTIFIER, "id"),
        Token(TokenType.EQUALS),
        Token(TokenType.INTEGER, 3),
        Token(TokenType.SEMICOLON),
        Token(TokenType.EOF),
    ]


def test_keywords_are_case_insensitive():
    # SQL keywords should be recognized regardless of case, but
    # identifiers should preserve their original casing exactly
    # as typed (e.g. table/column names aren't uppercased).
    tokens = tokenize("select * from Users;")

    assert tokens == [
        Token(TokenType.SELECT),
        Token(TokenType.ASTERISK),
        Token(TokenType.FROM),
        Token(TokenType.IDENTIFIER, "Users"),  # casing preserved
        Token(TokenType.SEMICOLON),
        Token(TokenType.EOF),
    ]


def test_multi_digit_integer():
    tokens = tokenize("WHERE id = 12345")
    assert Token(TokenType.INTEGER, 12345) in tokens


def test_empty_string_literal():
    tokens = tokenize("VALUES ('')")
    assert Token(TokenType.STRING, "") in tokens


def test_whitespace_variations_are_ignored():
    # Extra spaces, tabs, and newlines shouldn't affect tokenization.
    tokens = tokenize("SELECT   *\tFROM\nusers;")

    assert tokens == [
        Token(TokenType.SELECT),
        Token(TokenType.ASTERISK),
        Token(TokenType.FROM),
        Token(TokenType.IDENTIFIER, "users"),
        Token(TokenType.SEMICOLON),
        Token(TokenType.EOF),
    ]


def test_unterminated_string_raises_tokenizer_error():
    with pytest.raises(TokenizerError):
        tokenize("INSERT INTO users VALUES (1, 'alice)")


def test_unrecognized_character_raises_tokenizer_error():
    with pytest.raises(TokenizerError):
        tokenize("SELECT @ FROM users;")


def test_every_statement_ends_with_eof_token():
    tokens = tokenize("SELECT * FROM users;")
    assert tokens[-1] == Token(TokenType.EOF)