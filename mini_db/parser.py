"""
parser.py — Phase 3, step 2: turns a token list into an AST
(Abstract Syntax Tree).

    SQL text -> tokenizer.py: list of Tokens
             -> [THIS FILE: parser] -> AST (SelectStatement, etc.)
             -> executor.py: runs the AST against Table

Where the tokenizer only recognized the *shape* of individual pieces
of text (this is a keyword, this is a number), the parser is where
SQL grammar rules actually get enforced: a SELECT must have a FROM,
an INSERT must have VALUES with the right number of items, and so
on. If the tokens don't form a valid statement, the parser is what
raises an error — the tokenizer would have happily tokenized garbage
like "FROM FROM FROM" without complaint, since each word alone is a
valid token.

This is a *recursive-descent* parser: one method per grammar rule
(parse_select, parse_insert, parse_delete), each consuming tokens in
order and calling into other rule-methods where the grammar says to.
It's the same technique used by most hand-written parsers (including
CPython's own parser) because the code structure mirrors the grammar
structure directly — makes it easy to read a rule's method and see
exactly what it expects, in what order.
"""

from dataclasses import dataclass
from mini_db.tokenizer import Token, TokenType


# --- AST node types -------------------------------------------------
#
# Each dataclass below represents one fully-parsed SQL statement.
# These are what the executor will eventually consume instead of raw
# tokens — e.g. "run this SelectStatement" rather than "here's a list
# of tokens, figure out what they mean."
#
# where_column/where_value are generic (not hardcoded to "id") even
# though today's Executor only supports filtering by id — keeping
# the AST generic here means the parser doesn't need to change later
# when the executor supports filtering by other columns; only the
# executor's wiring code needs to grow.

@dataclass
class SelectStatement:
    table: str
    where_column: str | None = None
    where_value: object = None


@dataclass
class InsertStatement:
    table: str
    values: tuple  # (id, name) for now, per the hardcoded schema


@dataclass
class DeleteStatement:
    table: str
    where_column: str | None = None
    where_value: object = None


class ParserError(Exception):
    """Raised when the token sequence doesn't match any valid
    statement shape — e.g. a SELECT missing its FROM clause."""
    pass


class Parser:
    """
    Walks a token list left to right with a single position cursor,
    matching it against the grammar above. Each `parse_*` method
    assumes it's being called with the cursor sitting exactly where
    that rule should start, and leaves the cursor exactly where the
    next rule (or EOF) should start.
    """

    def __init__(self, tokens: list[Token]):
        self.tokens = tokens
        self.pos = 0

    # --- cursor helpers ------------------------------------------------

    def _peek(self) -> Token:
        return self.tokens[self.pos]

    def _advance(self) -> Token:
        token = self.tokens[self.pos]
        self.pos += 1
        return token

    def _expect(self, token_type: TokenType) -> Token:
        """
        Consume the next token only if it matches the expected type;
        otherwise raise a ParserError describing what was expected
        vs. what was actually found. Centralizing this check here
        (rather than repeating `if token.type != X: raise ...`
        everywhere) keeps every grammar rule below short and readable.
        """
        token = self._peek()
        if token.type != token_type:
            raise ParserError(
                f"Expected {token_type.name}, got {token.type.name} "
                f"at token position {self.pos}"
            )
        return self._advance()

    # --- entry point -----------------------------------------------------

    def parse(self):
        """
        Dispatches to the right statement parser based on the first
        token, then confirms every token was consumed (via the EOF
        check inside each parse_* method) so trailing garbage like
        "SELECT * FROM users EXTRA_JUNK" is rejected rather than
        silently ignored.
        """
        token = self._peek()

        if token.type == TokenType.SELECT:
            return self.parse_select()
        elif token.type == TokenType.INSERT:
            return self.parse_insert()
        elif token.type == TokenType.DELETE:
            return self.parse_delete()
        else:
            raise ParserError(
                f"Expected a statement to start with SELECT, INSERT, "
                f"or DELETE, got {token.type.name}"
            )

    # --- statement rules -------------------------------------------------

    def parse_select(self) -> SelectStatement:
        self._expect(TokenType.SELECT)
        self._expect(TokenType.ASTERISK)
        self._expect(TokenType.FROM)
        table = self._expect(TokenType.IDENTIFIER).value

        where_column, where_value = self._parse_optional_where()
        self._consume_optional_semicolon()
        self._expect(TokenType.EOF)

        return SelectStatement(table, where_column, where_value)

    def parse_insert(self) -> InsertStatement:
        self._expect(TokenType.INSERT)
        self._expect(TokenType.INTO)
        table = self._expect(TokenType.IDENTIFIER).value
        self._expect(TokenType.VALUES)
        self._expect(TokenType.LPAREN)

        row_id = self._expect(TokenType.INTEGER).value
        self._expect(TokenType.COMMA)
        name = self._expect(TokenType.STRING).value

        self._expect(TokenType.RPAREN)
        self._consume_optional_semicolon()
        self._expect(TokenType.EOF)

        return InsertStatement(table, (row_id, name))

    def parse_delete(self) -> DeleteStatement:
        self._expect(TokenType.DELETE)
        self._expect(TokenType.FROM)
        table = self._expect(TokenType.IDENTIFIER).value

        where_column, where_value = self._parse_optional_where()
        self._consume_optional_semicolon()
        self._expect(TokenType.EOF)

        return DeleteStatement(table, where_column, where_value)

    # --- shared sub-rules --------------------------------------------------

    def _parse_optional_where(self):
        """
        Shared by SELECT and DELETE, since both support:
            WHERE <column> = <literal>
        or no WHERE clause at all. Returns (None, None) if there's
        no WHERE, otherwise (column_name, value).
        """
        if self._peek().type != TokenType.WHERE:
            return None, None

        self._advance()  # consume WHERE
        column = self._expect(TokenType.IDENTIFIER).value
        self._expect(TokenType.EQUALS)

        value_token = self._peek()
        if value_token.type not in (TokenType.INTEGER, TokenType.STRING):
            raise ParserError(
                f"Expected a value (integer or string) after '=', "
                f"got {value_token.type.name}"
            )
        value = self._advance().value

        return column, value

    def _consume_optional_semicolon(self):
        if self._peek().type == TokenType.SEMICOLON:
            self._advance()


def parse(tokens: list[Token]):
    """
    Convenience function so callers don't need to instantiate Parser
    themselves for the common case of "parse this token list once."
    """
    return Parser(tokens).parse()