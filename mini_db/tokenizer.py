"""
tokenizer.py — Phase 3, step 1: turns raw SQL text into a flat list
of tokens.

This is the first stage of the SQL front end:

    SQL text -> [THIS FILE: tokenizer] -> list of Tokens
             -> parser.py: recursive-descent parser -> AST
             -> executor.py: runs the AST against Table

The tokenizer has no concept of SQL grammar. It doesn't know a
SELECT needs a FROM, or that WHERE takes a condition — it only knows
how to recognize the *shape* of individual pieces of text: "this
looks like a keyword," "this looks like a number," "this is a
quoted string." Whether those pieces form a valid statement is
entirely the parser's job, one file over.

No libraries beyond the standard-library `dataclasses` and `enum`
are used — writing this by hand (rather than using a lexer-generator
library) is intentional, per the project's learning goals.
"""

from dataclasses import dataclass
from enum import Enum, auto


class TokenType(Enum):
    # Keywords — case-insensitive in SQL, so "select" and "SELECT"
    # both become a SELECT token. Kept as distinct members (rather
    # than one generic KEYWORD type carrying a string) so the parser
    # can pattern-match on token *type* instead of comparing strings
    # everywhere, which is both faster and catches typos at parse time.
    SELECT = auto()
    INSERT = auto()
    INTO = auto()
    VALUES = auto()
    FROM = auto()
    WHERE = auto()
    DELETE = auto()

    IDENTIFIER = auto()   # e.g. table or column names: users, id, name
    INTEGER = auto()       # e.g. 3, 42
    STRING = auto()         # e.g. 'alice' (quotes stripped, value stored)

    ASTERISK = auto()    # *
    COMMA = auto()         # ,
    LPAREN = auto()        # (
    RPAREN = auto()        # )
    EQUALS = auto()         # =
    SEMICOLON = auto()   # ;

    EOF = auto()  # marks the end of input; lets the parser check
                  # "did we consume everything" without index math


# Reserved words mapped to their token type. Kept as a dict rather
# than a chain of if/elif so adding a new keyword later (e.g. AND,
# CREATE) is a one-line change here, not a new branch somewhere else.
KEYWORDS = {
    "SELECT": TokenType.SELECT,
    "INSERT": TokenType.INSERT,
    "INTO": TokenType.INTO,
    "VALUES": TokenType.VALUES,
    "FROM": TokenType.FROM,
    "WHERE": TokenType.WHERE,
    "DELETE": TokenType.DELETE,
}


@dataclass
class Token:
    """
    A single recognized chunk of SQL text.

    type  -> what kind of token this is (see TokenType)
    value -> the underlying Python value:
               - IDENTIFIER: the name as typed, e.g. "users"
               - INTEGER: an actual Python int, e.g. 3 (not the string "3" —
                 converting at tokenize time means the parser and executor
                 never have to do this conversion themselves)
               - STRING: the text inside the quotes, quotes stripped
               - everything else: usually None, since the type alone
                 (e.g. SEMICOLON) says everything needed
    """
    type: TokenType
    value: object = None

    def __repr__(self):
        if self.value is not None:
            return f"Token({self.type.name}, {self.value!r})"
        return f"Token({self.type.name})"


class TokenizerError(Exception):
    """Raised when the tokenizer hits text it doesn't recognize —
    e.g. an unterminated string, or a character like '@' that has
    no meaning in this grammar."""
    pass


def tokenize(sql: str) -> list[Token]:
    """
    Convert a full SQL statement string into a list of Tokens,
    ending with a single EOF token.

    This is a hand-written scanner: a single index `i` walks through
    `sql` one character at a time (occasionally jumping ahead when a
    multi-character token like an identifier or number is found).
    This mirrors how real lexers work, just without the performance
    tricks (e.g. regex compilation, DFA tables) a production database
    would use.
    """
    tokens: list[Token] = []
    i = 0
    length = len(sql)

    while i < length:
        char = sql[i]

        # --- whitespace: skip entirely, it carries no meaning ---
        if char.isspace():
            i += 1
            continue

        # --- single-character punctuation ---
        if char == "*":
            tokens.append(Token(TokenType.ASTERISK))
            i += 1
            continue

        if char == ",":
            tokens.append(Token(TokenType.COMMA))
            i += 1
            continue

        if char == "(":
            tokens.append(Token(TokenType.LPAREN))
            i += 1
            continue

        if char == ")":
            tokens.append(Token(TokenType.RPAREN))
            i += 1
            continue

        if char == "=":
            tokens.append(Token(TokenType.EQUALS))
            i += 1
            continue

        if char == ";":
            tokens.append(Token(TokenType.SEMICOLON))
            i += 1
            continue

        # --- quoted string literal, e.g. 'alice' ---
        if char == "'":
            end = sql.find("'", i + 1)
            if end == -1:
                raise TokenizerError(
                    f"Unterminated string literal starting at position {i}"
                )
            value = sql[i + 1:end]
            tokens.append(Token(TokenType.STRING, value))
            i = end + 1
            continue

        # --- integer literal, e.g. 42 ---
        if char.isdigit():
            start = i
            while i < length and sql[i].isdigit():
                i += 1
            tokens.append(Token(TokenType.INTEGER, int(sql[start:i])))
            continue

        # --- identifier or keyword, e.g. users, id, SELECT ---
        # SQL identifiers start with a letter or underscore, and may
        # contain letters, digits, or underscores after that.
        if char.isalpha() or char == "_":
            start = i
            while i < length and (sql[i].isalnum() or sql[i] == "_"):
                i += 1
            word = sql[start:i]

            keyword_type = KEYWORDS.get(word.upper())
            if keyword_type is not None:
                tokens.append(Token(keyword_type))
            else:
                tokens.append(Token(TokenType.IDENTIFIER, word))
            continue

        # --- anything else is unrecognized ---
        raise TokenizerError(f"Unexpected character {char!r} at position {i}")

    tokens.append(Token(TokenType.EOF))
    return tokens