#!/usr/bin/env python3
"""
Table Parser for MCP
Extracts Markdown/CSV tables from prompts and creates temporary SQLite databases
"""

import re
import pandas as pd
import sqlite3
import os
import tempfile


def extract_markdown_tables(text: str) -> list[tuple[str, pd.DataFrame]]:
    """
    Extract Markdown tables from text

    Args:
        text: Text containing Markdown tables

    Returns:
        List of (table_name, dataframe) tuples
    """
    tables = []

    # Find all Markdown tables in the text
    # Pattern: lines starting with | ... |
    lines = text.split('\n')
    current_table = []
    table_count = 0

    for line in lines:
        line = line.strip()
        if line.startswith('|') and line.endswith('|'):
            current_table.append(line)
        else:
            if current_table:
                # End of table, parse it
                try:
                    df = parse_markdown_table(current_table)
                    if not df.empty:
                        table_name = f"table_{table_count}"
                        tables.append((table_name, df))
                        table_count += 1
                except Exception as e:
                    print(f"Warning: Could not parse table: {e}")
                current_table = []

    # Handle last table if exists
    if current_table:
        try:
            df = parse_markdown_table(current_table)
            if not df.empty:
                table_name = f"table_{table_count}"
                tables.append((table_name, df))
        except Exception:
            pass

    return tables


def parse_markdown_table(lines: list[str]) -> pd.DataFrame:
    """
    Parse Markdown table lines into DataFrame (ROBUST VERSION)

    Args:
        lines: List of table lines (|col1|col2|...)

    Returns:
        DataFrame

    Handles:
    - Column count mismatches (padding/truncating)
    - Duplicate column names (auto-renaming)
    - Empty column names (auto-naming)
    """
    if len(lines) < 2:
        return pd.DataFrame()

    # Remove separator line (|:---|:---|)
    data_lines = [lines[0]]  # Header
    for line in lines[1:]:
        if not re.match(r'^\|[\s\:\-]+\|', line):
            data_lines.append(line)

    if len(data_lines) < 2:  # Need at least header + 1 row
        return pd.DataFrame()

    # Parse directly into DataFrame (avoid CSV conversion issues)
    # Extract header
    header_line = data_lines[0].strip('|')
    header = [cell.strip() for cell in header_line.split('|')]

    # FIX 1: Handle duplicate column names
    seen_names = {}
    unique_header = []
    for col_name in header:
        # Handle empty column names
        if not col_name:
            col_name = "unnamed"

        # Handle duplicates
        if col_name in seen_names:
            seen_names[col_name] += 1
            unique_header.append(f"{col_name}_{seen_names[col_name]}")
        else:
            seen_names[col_name] = 0
            unique_header.append(col_name)

    header = unique_header
    num_cols = len(header)

    # Extract rows
    rows = []
    for line in data_lines[1:]:
        line = line.strip('|')
        row = [cell.strip() for cell in line.split('|')]

        # FIX 2: Handle column count mismatch
        if len(row) < num_cols:
            # Pad with empty strings
            row.extend([''] * (num_cols - len(row)))
        elif len(row) > num_cols:
            # Truncate extra columns
            row = row[:num_cols]

        rows.append(row)

    try:
        df = pd.DataFrame(rows, columns=header)
        return df
    except Exception as e:
        print(f"Error parsing table: {e}")
        return pd.DataFrame()


def create_temp_database(tables: list[tuple[str, pd.DataFrame]]) -> str:
    """
    Create temporary SQLite database from tables

    Args:
        tables: List of (table_name, dataframe) tuples

    Returns:
        Path to temporary database file
    """
    # Create temporary file
    fd, db_path = tempfile.mkstemp(suffix='.sqlite', prefix='mmtu_mcp_')
    os.close(fd)

    # Create database
    conn = sqlite3.connect(db_path)

    for table_name, df in tables:
        try:
            df.to_sql(table_name, conn, index=False, if_exists='replace')
        except Exception as e:
            print(f"Warning: Could not create table {table_name}: {e}")

    conn.close()

    return db_path


def extract_and_create_database(prompt: str, metadata: dict) -> str | None:
    """
    Extract tables from prompt and create temporary database

    Args:
        prompt: The prompt text (may contain Markdown tables)
        metadata: Metadata dict (may contain sqlite_path for NL2SQL)

    Returns:
        Database path or None if no tables found
    """
    print(f"[TABLE PARSER] extract_and_create_database called")

    # Check if we already have a sqlite_path (NL2SQL tasks)
    if 'sqlite_path' in metadata:
        sqlite_path = metadata['sqlite_path']
        print(f"[TABLE PARSER] Found sqlite_path in metadata: {sqlite_path}")
        # Expand $MMTU_HOME
        if '$MMTU_HOME' in sqlite_path:
            sqlite_path = sqlite_path.replace(
                '$MMTU_HOME',
                os.environ.get('MMTU_HOME', '/app')
            )
        if os.path.exists(sqlite_path):
            print(f"[TABLE PARSER] Using existing database: {sqlite_path}")
            return sqlite_path
        else:
            print(f"[TABLE PARSER] Database not found: {sqlite_path}")

    # Extract tables from prompt
    print(f"[TABLE PARSER] Extracting tables from prompt ({len(prompt)} chars)...")
    tables = extract_markdown_tables(prompt)
    print(f"[TABLE PARSER] Found {len(tables)} tables")

    if not tables:
        print(f"[TABLE PARSER] No tables found, returning None")
        return None

    # Create temporary database
    print(f"[TABLE PARSER] Creating temp database...")
    db_path = create_temp_database(tables)
    print(f"[TABLE PARSER] Database created: {db_path}")

    return db_path


def cleanup_temp_database(db_path: str):
    """
    Remove temporary database file

    Args:
        db_path: Path to database file
    """
    if db_path and os.path.exists(db_path) and 'mmtu_mcp_' in db_path:
        try:
            os.remove(db_path)
        except Exception as e:
            print(f"Warning: Could not remove temp database: {e}")


if __name__ == "__main__":
    # Test
    test_prompt = """
    Here is a table:

    | name | age | city |
    |:-----|:----|:-----|
    | Alice | 30 | NYC |
    | Bob | 25 | LA |
    | Charlie | 35 | SF |

    What is the average age?
    """

    tables = extract_markdown_tables(test_prompt)
    print(f"Found {len(tables)} tables")

    if tables:
        for name, df in tables:
            print(f"\nTable: {name}")
            print(df)

        db_path = create_temp_database(tables)
        print(f"\nCreated database: {db_path}")

        # Test query
        conn = sqlite3.connect(db_path)
        cursor = conn.execute("SELECT * FROM table_0")
        print("\nQuery result:")
        print(cursor.fetchall())
        conn.close()

        cleanup_temp_database(db_path)
        print("Cleaned up")
