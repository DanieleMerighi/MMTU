#!/usr/bin/env python3
"""
MCP Server for Table Tools

Provides MCP-compliant tools for database exploration:
- get_schema: Returns table structure
- execute_sql: Executes SQL query with timeout
- sample_rows: Gets N sample rows
- count_rows: Returns total row count
- get_distinct_values: Returns unique values for column
- get_statistics: Returns min/max/avg for numeric columns
"""

import sqlite3
import pandas as pd
import os
from typing import Dict, List, Any, Optional
from pathlib import Path


class TableMCPServer:
    """
    MCP Server for table exploration tools

    This server provides tools that allow LLMs to explore and query databases
    without needing the full table in context.
    """

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize MCP server with optional database path

        Args:
            db_path: Path to SQLite database. Can be set later via connect()
        """
        self.conn = None
        self.db_path = None

        if db_path:
            self.connect(db_path)

    def connect(self, db_path: str):
        """
        Connect to SQLite database

        Args:
            db_path: Path to SQLite database (can contain $MMTU_HOME)
        """
        # Expand environment variables
        db_path = os.path.expandvars(db_path)

        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Database not found: {db_path}")

        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)

    def disconnect(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            self.conn = None

    def get_tools_schema(self) -> List[Dict[str, Any]]:
        """
        Returns MCP tool definitions compatible with Llama 3.2 function calling

        Returns:
            List of tool definitions in OpenAI function calling format
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": "get_schema",
                    "description": "Get the structure of all tables in the database, including column names and types",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "count_rows",
                    "description": "Get the total number of rows in a specific table",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "table_name": {
                                "type": "string",
                                "description": "Name of the table to count rows in"
                            }
                        },
                        "required": ["table_name"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "sample_rows",
                    "description": "Get a sample of N rows from a table to understand the data",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "table_name": {
                                "type": "string",
                                "description": "Name of the table to sample from"
                            },
                            "n": {
                                "type": "integer",
                                "description": "Number of rows to sample (default: 10, max: 100)",
                                "default": 10
                            }
                        },
                        "required": ["table_name"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "execute_sql",
                    "description": "Execute a SQL SELECT query on the database and return results. Use this to answer questions about the data.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "SQL SELECT query to execute"
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_distinct_values",
                    "description": "Get all unique values from a specific column",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "table_name": {
                                "type": "string",
                                "description": "Name of the table"
                            },
                            "column_name": {
                                "type": "string",
                                "description": "Name of the column to get distinct values from"
                            }
                        },
                        "required": ["table_name", "column_name"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_statistics",
                    "description": "Get basic statistics (min, max, avg) for a numeric column",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "table_name": {
                                "type": "string",
                                "description": "Name of the table"
                            },
                            "column_name": {
                                "type": "string",
                                "description": "Name of the numeric column"
                            }
                        },
                        "required": ["table_name", "column_name"]
                    }
                }
            }
        ]

    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """
        Execute a tool and return result as string

        Args:
            tool_name: Name of the tool to execute
            arguments: Tool arguments

        Returns:
            Tool result as formatted string
        """
        if not self.conn:
            return "Error: No database connected"

        try:
            if tool_name == "get_schema":
                return self.get_schema()
            elif tool_name == "count_rows":
                return self.count_rows(arguments.get("table_name"))
            elif tool_name == "sample_rows":
                return self.sample_rows(
                    arguments.get("table_name"),
                    arguments.get("n", 10)
                )
            elif tool_name == "execute_sql":
                return self.execute_sql(arguments.get("query"))
            elif tool_name == "get_distinct_values":
                return self.get_distinct_values(
                    arguments.get("table_name"),
                    arguments.get("column_name")
                )
            elif tool_name == "get_statistics":
                return self.get_statistics(
                    arguments.get("table_name"),
                    arguments.get("column_name")
                )
            else:
                return f"Error: Unknown tool '{tool_name}'"
        except Exception as e:
            return f"Error executing {tool_name}: {str(e)}"

    # Tool Implementations

    def get_schema(self) -> str:
        """Get database schema for all tables"""
        cursor = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        tables = [row[0] for row in cursor.fetchall()]

        schema_parts = []
        for table in tables:
            cursor = self.conn.execute(f"PRAGMA table_info({table})")
            columns = cursor.fetchall()

            schema_parts.append(f"Table: {table}")
            for col in columns:
                schema_parts.append(f"  - {col[1]} ({col[2]})")

        return "\n".join(schema_parts)

    def count_rows(self, table_name: str) -> str:
        """Count total rows in table"""
        cursor = self.conn.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = cursor.fetchone()[0]
        return f"Table '{table_name}' has {count} rows"

    def sample_rows(self, table_name: str, n: int = 10) -> str:
        """Get sample of N rows from table"""
        n = min(n, 100)  # Max 100 rows
        df = pd.read_sql(f"SELECT * FROM {table_name} LIMIT {n}", self.conn)
        return f"Sample of {len(df)} rows from '{table_name}':\n{df.to_markdown(index=False)}"

    def execute_sql(self, query: str, timeout: int = 15) -> str:
        """
        Execute SQL query with timeout

        Args:
            query: SQL query to execute
            timeout: Timeout in seconds (default: 15)

        Returns:
            Query results as markdown table
        """
        # Security: Only allow SELECT queries
        if not query.strip().upper().startswith("SELECT"):
            return "Error: Only SELECT queries are allowed"

        try:
            # Set timeout
            self.conn.execute(f"PRAGMA timeout = {timeout * 1000}")

            # Execute query
            df = pd.read_sql(query, self.conn)

            if len(df) == 0:
                return "Query returned 0 rows"

            # Limit output size
            if len(df) > 100:
                return f"Query returned {len(df)} rows (showing first 100):\n{df.head(100).to_markdown(index=False)}"

            return f"Query returned {len(df)} rows:\n{df.to_markdown(index=False)}"

        except Exception as e:
            return f"SQL Error: {str(e)}"

    def get_distinct_values(self, table_name: str, column_name: str) -> str:
        """Get distinct values from column"""
        cursor = self.conn.execute(
            f"SELECT DISTINCT {column_name} FROM {table_name} ORDER BY {column_name} LIMIT 100"
        )
        values = [row[0] for row in cursor.fetchall()]

        if len(values) > 100:
            return f"Column '{column_name}' has >100 distinct values. Showing first 100:\n" + "\n".join(map(str, values[:100]))

        return f"Distinct values in '{column_name}':\n" + "\n".join(map(str, values))

    def get_statistics(self, table_name: str, column_name: str) -> str:
        """Get statistics for numeric column"""
        try:
            cursor = self.conn.execute(
                f"SELECT MIN({column_name}), MAX({column_name}), AVG({column_name}) FROM {table_name}"
            )
            min_val, max_val, avg_val = cursor.fetchone()

            return f"Statistics for '{column_name}':\n" + \
                   f"  Min: {min_val}\n" + \
                   f"  Max: {max_val}\n" + \
                   f"  Avg: {avg_val:.2f}" if avg_val else "N/A"
        except Exception as e:
            return f"Error: Column '{column_name}' may not be numeric: {str(e)}"


def main():
    """Test the MCP server standalone"""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python table_mcp_server.py <db_path>")
        sys.exit(1)

    server = TableMCPServer(sys.argv[1])

    print("MCP Server Connected!")
    print("\nAvailable Tools:")
    for tool in server.get_tools_schema():
        print(f"  - {tool['function']['name']}: {tool['function']['description']}")

    print("\n" + "="*80)
    print("Testing Tools:")
    print("="*80)

    # Test get_schema
    print("\n1. get_schema:")
    print(server.get_schema())

    # Test count_rows (assume 'conference' table exists)
    print("\n2. count_rows (conference):")
    print(server.count_rows("conference"))

    # Test sample_rows
    print("\n3. sample_rows (conference, 5):")
    print(server.sample_rows("conference", 5))

    server.disconnect()


if __name__ == "__main__":
    main()
