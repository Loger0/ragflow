#
#  Copyright 2025 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
import json
import logging
import os
import re
import threading
import time
from abc import abstractmethod
from typing import Any

from sqlalchemy import Column, JSON, Table, Text
from sqlalchemy.dialects.postgresql import VARCHAR

from common.doc_store.doc_store_base import DocStoreConnection, MatchExpr, OrderByExpr

ATTEMPT_TIME = 2

# Common templates for Vastbase (PostgreSQL dialect)
# ADAPT: PostgreSQL LIMIT/OFFSET syntax replaces MySQL LIMIT offset,limit
# ADAPT: pgvector <=> operator replaces cosine_distance() for vector search
# ADAPT: bm25() / @~@ replaces MATCH() AGAINST() for fulltext search
index_name_template = "ix_%s_%s"
fulltext_index_name_template = "fts_idx_%s"
fulltext_search_template = "\"%s\" @~@ '%s'"  # Vastbase BM25 operator
vector_search_template = "%s <=> '%s'"
vector_column_pattern = re.compile(r"q_(?P<vector_size>\d+)_vec")

# Document metadata table columns (PostgreSQL types)
doc_meta_columns = [
    Column("id", VARCHAR(256), primary_key=True, comment="document id"),
    Column("kb_id", VARCHAR(256), nullable=False, comment="knowledge base id"),
    Column("meta_fields", JSON, nullable=True, comment="document metadata fields"),
]
doc_meta_column_names = [col.name for col in doc_meta_columns]
doc_meta_column_types = {col.name: col.type for col in doc_meta_columns}


def get_value_str(value: Any) -> str:
    """Convert value to PostgreSQL SQL string representation.

    Args:
        value: Value to convert to SQL string.

    Returns:
        SQL-safe string representation.
    """
    if isinstance(value, str):
        # ADAPT: Manual PG string escaping replaces pymysql escape_string
        escaped = value.replace("'", "''").replace("\\", "\\\\")
        return f"'{escaped}'"
    elif isinstance(value, bool):
        return "true" if value else "false"
    elif value is None:
        return "NULL"
    elif isinstance(value, (list, dict)):
        json_str = json.dumps(value, ensure_ascii=False)
        escaped = json_str.replace("'", "''")
        return f"'{escaped}'"
    else:
        return str(value)


def _try_with_lock(lock_name: str, process_func, check_func, timeout: int = None):
    """Execute function with distributed lock.

    Args:
        lock_name: Name of the distributed lock.
        process_func: Function to execute while holding the lock.
        check_func: Function to check if the operation was already completed.
        timeout: Maximum time to wait for completion.
    """
    if not timeout:
        timeout = int(os.environ.get("VB_DDL_TIMEOUT", "60"))

    if not check_func():
        from rag.utils.redis_conn import RedisDistributedLock
        lock = RedisDistributedLock(lock_name)
        if lock.acquire():
            try:
                process_func()
                return
            except Exception as e:
                if "Duplicate" in str(e) or "already exists" in str(e):
                    return
                raise
            finally:
                lock.release()

    if not check_func():
        time.sleep(1)
        count = 1
        while count < timeout and not check_func():
            count += 1
            time.sleep(1)
        if count >= timeout and not check_func():
            raise Exception(f"Timeout to wait for process complete for {lock_name}.")


class VastbaseConnectionBase(DocStoreConnection):
    """Base class for Vastbase document store connections.

    Independent from OBConnectionBase. Uses PostgreSQL/Vastbase SQL dialect
    and pyvastbase for connection management.

    Subclasses must implement the template methods:
        get_index_columns(), get_fulltext_columns(), get_column_definitions(),
        get_lock_prefix(), and all abstract CRUD methods from DocStoreConnection.
    """

    def __init__(self, logger_name: str = 'ragflow.vastbase_conn'):
        from common.doc_store.vastbase_conn_pool import VB_CONN

        self.logger = logging.getLogger(logger_name)
        self.client = VB_CONN.get_client()
        self.db_name = VB_CONN.get_db_name()
        self.uri = VB_CONN.get_uri()

        self._load_env_vars()

        # ADAPT: Table existence cache uses set[str] for fast lookup
        self._table_exists_cache: set[str] = set()
        self._table_exists_cache_lock = threading.RLock()

        # Cache for vector columns: stores (table_name, vector_size) tuples
        self._vector_column_cache: set[tuple[str, int]] = set()
        self._vector_column_cache_lock = threading.RLock()

        self.logger.info(f"Vastbase {self.uri} connection initialized.")

    def _load_env_vars(self):
        def is_true(var: str, default: str) -> bool:
            return os.getenv(var, default).lower() in ['true', '1', 'yes', 'y']

        self.enable_fulltext_search = is_true('ENABLE_FULLTEXT_SEARCH', 'true')
        self.use_fulltext_hint = is_true('USE_FULLTEXT_HINT', 'true')
        self.search_original_content = is_true("SEARCH_ORIGINAL_CONTENT", 'true')
        self.enable_hybrid_search = is_true('ENABLE_HYBRID_SEARCH', 'false')
        self.use_fulltext_first_fusion_search = is_true('USE_FULLTEXT_FIRST_FUSION_SEARCH', 'true')

    def _execute_sql(self, sql: str, params: tuple = None):
        """Execute a raw SQL statement and return all rows.

        Uses cursor.description to distinguish DML statements (INSERT/UPDATE/DELETE)
        from queries (SELECT).  DML statements return an empty list after commit;
        queries return fetched rows.

        Args:
            sql: SQL statement to execute.
            params: Optional query parameters.

        Returns:
            List of result rows (empty list for DML statements).
        """
        try:
            cursor = self.client.cursor()
            cursor.execute(sql, params)
            # ADAPT: Use DB-API 2.0 cursor.description to distinguish DML from queries.
            # cursor.description is None for statements that produce no result set
            # (INSERT, UPDATE, DELETE, CREATE, ALTER, etc.).
            if cursor.description is None:
                self.client.commit()
                return []
            return cursor.fetchall()
        except Exception as e:
            self.logger.error(f"SQL execution error: {str(e)}, SQL: {sql}")
            try:
                self.client.rollback()
            except Exception:
                pass
            raise

    """
    Template methods - must be implemented by subclasses
    """

    @abstractmethod
    def get_index_columns(self) -> list[str]:
        """Return list of column names that need regular indexes."""
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def get_fulltext_columns(self) -> list[str]:
        """Return list of column names that need fulltext indexes (without weight suffix)."""
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def get_column_definitions(self) -> list:
        """Return list of column definitions for table creation."""
        raise NotImplementedError("Not implemented")

    def get_extra_columns(self) -> list:
        """Return list of extra columns to add after table creation. Override if needed."""
        return []

    def get_table_name(self, index_name: str, dataset_id: str) -> str:
        """Return the actual table name given index_name and dataset_id."""
        return index_name

    @abstractmethod
    def get_lock_prefix(self) -> str:
        """Return the lock name prefix for distributed locking."""
        raise NotImplementedError("Not implemented")

    """
    Database operations
    """

    def db_type(self) -> str:
        # ADAPT: Return "vastbase" instead of "oceanbase"
        return "vastbase"

    def health(self) -> dict:
        # ADAPT: SELECT version() replaces SHOW VARIABLES LIKE 'version_comment'
        try:
            rows = self._execute_sql("SELECT version()")
            version_str = rows[0][0] if rows else "unknown"
            return {
                "uri": self.uri,
                "version": version_str,
            }
        except Exception as e:
            return {
                "uri": self.uri,
                "version": f"error: {str(e)}",
            }

    """
    Table operations - common implementation using template methods
    """

    def _check_table_exists_cached(self, table_name: str) -> bool:
        """Check table existence with cache to reduce INFORMATION_SCHEMA queries.

        Thread-safe implementation using RLock.

        Args:
            table_name: Name of the table to check.

        Returns:
            True if table exists with all required indexes, False otherwise.
        """
        if table_name in self._table_exists_cache:
            return True

        try:
            # ADAPT: PostgreSQL pg_tables replaces ObVecClient.check_table_exists()
            sql = (
                "SELECT EXISTS ("
                "  SELECT 1 FROM information_schema.tables "
                "  WHERE table_schema = 'public' AND table_name = %s"
                ")"
            )
            rows = self._execute_sql(sql, (table_name,))
            if not rows or not rows[0][0]:
                return False

            # Check regular indexes
            for column_name in self.get_index_columns():
                idx_name = index_name_template % (table_name, column_name)
                if not self._index_exists(table_name, idx_name):
                    return False

            # Check fulltext indexes
            for column_name in self.get_fulltext_columns():
                fts_idx_name = fulltext_index_name_template % column_name
                if not self._index_exists(table_name, fts_idx_name):
                    return False

            # Check extra columns
            for column in self.get_extra_columns():
                if not self._column_exist(table_name, column.name):
                    return False

        except Exception as e:
            raise Exception(f"VastbaseConnectionBase._check_table_exists_cached error: {str(e)}")

        with self._table_exists_cache_lock:
            if table_name not in self._table_exists_cache:
                self._table_exists_cache.add(table_name)
        return True

    def _table_exists(self, table_name: str) -> bool:
        """Check if a table exists in the database.

        Args:
            table_name: Name of the table.

        Returns:
            True if table exists.
        """
        sql = (
            "SELECT EXISTS ("
            "  SELECT 1 FROM information_schema.tables "
            "  WHERE table_schema = 'public' AND table_name = %s"
            ")"
        )
        rows = self._execute_sql(sql, (table_name,))
        return bool(rows and rows[0][0])

    def _create_table(self, table_name: str):
        """Create table using column definitions from subclass.

        Args:
            table_name: Name of the table to create.
        """
        self._create_table_with_columns(table_name, self.get_column_definitions())

    def create_idx(self, index_name: str, dataset_id: str, vector_size: int, parser_id: str = None):
        """Create index/table with all necessary indexes.

        Args:
            index_name: Base name for the index/table.
            dataset_id: Dataset identifier.
            vector_size: Size of vector columns.
            parser_id: Optional parser identifier.
        """
        table_name = self.get_table_name(index_name, dataset_id)
        lock_prefix = self.get_lock_prefix()

        try:
            _try_with_lock(
                lock_name=f"{lock_prefix}create_table_{table_name}",
                check_func=lambda: self._table_exists(table_name),
                process_func=lambda: self._create_table(table_name),
            )

            for column_name in self.get_index_columns():
                _try_with_lock(
                    lock_name=f"{lock_prefix}add_idx_{table_name}_{column_name}",
                    check_func=lambda cn=column_name: self._index_exists(
                        table_name, index_name_template % (table_name, cn)
                    ),
                    process_func=lambda cn=column_name: self._add_index(table_name, cn),
                )

            for column_name in self.get_fulltext_columns():
                _try_with_lock(
                    lock_name=f"{lock_prefix}add_fulltext_idx_{table_name}_{column_name}",
                    check_func=lambda cn=column_name: self._index_exists(
                        table_name, fulltext_index_name_template % cn
                    ),
                    process_func=lambda cn=column_name: self._add_fulltext_index(table_name, cn),
                )

            # Add vector column and index
            self._ensure_vector_column_exists(table_name, vector_size)

            # Add extra columns if any
            for column in self.get_extra_columns():
                _try_with_lock(
                    lock_name=f"{lock_prefix}add_{column.name}_{table_name}",
                    check_func=lambda c=column: self._column_exist(table_name, c.name),
                    process_func=lambda c=column: self._add_column(table_name, c),
                )

        except Exception as e:
            raise Exception(f"VastbaseConnectionBase.create_idx error: {str(e)}")

    def create_doc_meta_idx(self, index_name: str):
        """Create a document metadata table.

        Table name pattern: ragflow_doc_meta_{tenant_id}

        Args:
            index_name: Table name for the metadata table.

        Returns:
            True on success, False on failure.
        """
        table_name = index_name
        lock_prefix = self.get_lock_prefix()

        try:
            _try_with_lock(
                lock_name=f"{lock_prefix}create_doc_meta_table_{table_name}",
                check_func=lambda: self._table_exists(table_name),
                process_func=lambda: self._create_table_with_columns(table_name, doc_meta_columns),
            )

            _try_with_lock(
                lock_name=f"{lock_prefix}add_idx_{table_name}_kb_id",
                check_func=lambda: self._index_exists(
                    table_name, index_name_template % (table_name, "kb_id")
                ),
                process_func=lambda: self._add_index(table_name, "kb_id"),
            )

            self.logger.info(f"Created document metadata table '{table_name}'.")
            return True

        except Exception as e:
            self.logger.error(f"VastbaseConnectionBase.create_doc_meta_idx error: {str(e)}")
            return False

    def delete_idx(self, index_name: str, dataset_id: str):
        """Delete index/table.

        Args:
            index_name: Base name of the table.
            dataset_id: Dataset identifier.
        """
        if index_name.startswith("ragflow_doc_meta_"):
            table_name = index_name
        else:
            table_name = self.get_table_name(index_name, dataset_id)
        try:
            if self._table_exists(table_name):
                self._execute_sql(f"DROP TABLE IF EXISTS \"{table_name}\"")
                self.logger.info(f"Dropped table '{table_name}'.")
        except Exception as e:
            raise Exception(f"VastbaseConnectionBase.delete_idx error: {str(e)}")

    def index_exist(self, index_name: str, dataset_id: str = None) -> bool:
        """Check if index/table exists.

        Args:
            index_name: Base name of the table.
            dataset_id: Optional dataset identifier.

        Returns:
            True if the table and all required indexes exist.
        """
        if index_name.startswith("ragflow_doc_meta_"):
            if index_name in self._table_exists_cache:
                return True
            if not self._table_exists(index_name):
                return False
            with self._table_exists_cache_lock:
                self._table_exists_cache.add(index_name)
            return True
        table_name = self.get_table_name(index_name, dataset_id) if dataset_id else index_name
        return self._check_table_exists_cached(table_name)

    """
    Table operations - helper methods
    """

    def _get_count(self, table_name: str, filter_list: list[str] = None,
                   filter_params: tuple = None) -> int:
        where_clause = ""
        if filter_list and len(filter_list) > 0:
            where_clause = "WHERE " + " AND ".join(filter_list)
        rows = self._execute_sql(
            f"SELECT COUNT(*) FROM {table_name} {where_clause}",
            filter_params,
        )
        return rows[0][0] if rows else 0

    def _column_exist(self, table_name: str, column_name: str) -> bool:
        # ADAPT: PostgreSQL information_schema.columns replaces MySQL INFORMATION_SCHEMA.COLUMNS
        sql = (
            "SELECT EXISTS ("
            "  SELECT 1 FROM information_schema.columns "
            "  WHERE table_schema = 'public' AND table_name = %s AND column_name = %s"
            ")"
        )
        rows = self._execute_sql(sql, (table_name, column_name))
        return bool(rows and rows[0][0])

    def _index_exists(self, table_name: str, idx_name: str) -> bool:
        # ADAPT: PostgreSQL pg_indexes replaces MySQL INFORMATION_SCHEMA.STATISTICS
        sql = (
            "SELECT EXISTS ("
            "  SELECT 1 FROM pg_indexes "
            "  WHERE schemaname = 'public' AND tablename = %s AND indexname = %s"
            ")"
        )
        rows = self._execute_sql(sql, (table_name, idx_name))
        return bool(rows and rows[0][0])

    def _create_table_with_columns(self, table_name: str, columns: list):
        """Create table with specified columns using raw PostgreSQL SQL.

        Args:
            table_name: Name of the table to create.
            columns: List of SQLAlchemy Column objects.
        """
        column_defs = []
        primary_keys = []
        for col in columns:
            col_type = self._map_column_type(col)
            nullable = "NOT NULL" if not col.nullable else ""
            if col.primary_key:
                primary_keys.append(col.name)
                nullable = "NOT NULL"
            comment = f" -- {col.comment}" if hasattr(col, 'comment') and col.comment else ""
            column_defs.append(f"\"{col.name}\" {col_type} {nullable}{comment}")

        if primary_keys:
            column_defs.append(f"PRIMARY KEY ({', '.join(primary_keys)})")

        create_sql = f"CREATE TABLE IF NOT EXISTS \"{table_name}\" (\n  " + ",\n  ".join(column_defs) + "\n)"
        self._execute_sql(create_sql)
        self.logger.info(f"Created table '{table_name}'.")

    def _map_column_type(self, col) -> str:
        """Map SQLAlchemy column type to PostgreSQL type string.

        Args:
            col: SQLAlchemy Column object.

        Returns:
            PostgreSQL type string.
        """
        from sqlalchemy import Integer, String, Float, JSON, Text
        from sqlalchemy.dialects.postgresql import ARRAY, VARCHAR

        col_type = col.type
        if isinstance(col_type, VARCHAR):
            return f"VARCHAR({col_type.length})"
        elif isinstance(col_type, String):
            if hasattr(col_type, 'length') and col_type.length:
                return f"VARCHAR({col_type.length})"
            return "TEXT"
        elif isinstance(col_type, Integer):
            return "INTEGER"
        elif isinstance(col_type, Float):
            return "DOUBLE PRECISION"
        elif isinstance(col_type, JSON):
            return "JSONB"
        elif isinstance(col_type, Text):
            return "TEXT"
        elif isinstance(col_type, ARRAY):
            inner = self._map_array_inner_type(col_type)
            return f"{inner}[]"
        else:
            # Default fallback
            return "TEXT"

    def _map_array_inner_type(self, array_type) -> str:
        """Map inner type of ARRAY column to PostgreSQL type string.

        Args:
            array_type: SQLAlchemy ARRAY type.

        Returns:
            PostgreSQL inner type string.
        """
        from sqlalchemy import Integer, String
        from sqlalchemy.dialects.postgresql import VARCHAR

        inner = array_type.item_type
        if isinstance(inner, VARCHAR):
            return f"VARCHAR({inner.length})"
        elif isinstance(inner, String):
            if hasattr(inner, 'length') and inner.length:
                return f"VARCHAR({inner.length})"
            return "TEXT"
        elif isinstance(inner, Integer):
            return "INTEGER"
        else:
            return "TEXT"

    def _add_index(self, table_name: str, column_name: str):
        idx_name = index_name_template % (table_name, column_name)
        # ADAPT: PostgreSQL CREATE INDEX replaces ObVecClient.create_index()
        self._execute_sql(
            f"CREATE INDEX IF NOT EXISTS \"{idx_name}\" ON \"{table_name}\" (\"{column_name}\")"
        )
        self.logger.info(f"Created index '{idx_name}' on table '{table_name}'.")

    def _add_fulltext_index(self, table_name: str, column_name: str):
        # ADAPT: Vastbase BM25 fulltext index replaces OceanBase FTS index
        fulltext_idx_name = fulltext_index_name_template % column_name
        try:
            # Use pyvastbase create_fulltext_index for BM25 support
            from pyvastbase import create_fulltext_index
            create_fulltext_index(
                collection_name=table_name,
                field_name=column_name,
                index_name=fulltext_idx_name,
                using="vastbase_doc_store",
            )
        except Exception:
            # Fallback: create via raw SQL for Vastbase
            self._execute_sql(
                f"CREATE INDEX IF NOT EXISTS \"{fulltext_idx_name}\" "
                f"ON \"{table_name}\" USING bm25 (\"{column_name}\")"
            )
        self.logger.info(
            f"Created fulltext index '{fulltext_idx_name}' on table '{table_name}'."
        )

    def _add_vector_column(self, table_name: str, vector_size: int):
        vector_field_name = f"q_{vector_size}_vec"
        # ADAPT: pgvector vector type replaces OceanBase VECTOR type
        self._execute_sql(
            f"ALTER TABLE \"{table_name}\" ADD COLUMN IF NOT EXISTS "
            f"\"{vector_field_name}\" vector({vector_size})"
        )
        self.logger.info(
            f"Added vector column '{vector_field_name}' to table '{table_name}'."
        )

    def _add_vector_index(self, table_name: str, vector_field_name: str):
        vector_idx_name = f"{vector_field_name}_idx"
        # ADAPT: pgvector HNSW index replaces OceanBase vsag vector index
        self._execute_sql(
            f"CREATE INDEX IF NOT EXISTS \"{vector_idx_name}\" "
            f"ON \"{table_name}\" USING hnsw (\"{vector_field_name}\" vector_cosine_ops)"
        )
        self.logger.info(
            f"Created vector index '{vector_idx_name}' on table '{table_name}' "
            f"with column '{vector_field_name}'."
        )

    def _add_column(self, table_name: str, column: object):
        """Add a column to an existing table.

        Args:
            table_name: Name of the table.
            column: SQLAlchemy Column object.
        """
        col_type = self._map_column_type(column)
        try:
            self._execute_sql(
                f"ALTER TABLE \"{table_name}\" ADD COLUMN IF NOT EXISTS "
                f"\"{column.name}\" {col_type}"
            )
            self.logger.info(f"Added column '{column.name}' to table '{table_name}'.")
        except Exception as e:
            self.logger.warning(
                f"Failed to add column '{column.name}' to table '{table_name}': {str(e)}"
            )

    def _ensure_vector_column_exists(self, table_name: str, vector_size: int):
        """Ensure vector column and index exist for the given vector size.

        Safe to call multiple times - skips if already exists.
        Uses cache to avoid repeated queries.

        Args:
            table_name: Name of the table.
            vector_size: Size of the vector column.
        """
        if vector_size <= 0:
            return

        cache_key = (table_name, vector_size)

        if cache_key in self._vector_column_cache:
            return

        lock_prefix = self.get_lock_prefix()
        vector_field_name = f"q_{vector_size}_vec"
        vector_index_name = f"{vector_field_name}_idx"

        column_exists = self._column_exist(table_name, vector_field_name)
        index_exists = self._index_exists(table_name, vector_index_name)

        if column_exists and index_exists:
            with self._vector_column_cache_lock:
                self._vector_column_cache.add(cache_key)
            return

        if not column_exists:
            _try_with_lock(
                lock_name=f"{lock_prefix}add_vector_column_{table_name}_{vector_field_name}",
                check_func=lambda: self._column_exist(table_name, vector_field_name),
                process_func=lambda: self._add_vector_column(table_name, vector_size),
            )

        if not index_exists:
            _try_with_lock(
                lock_name=f"{lock_prefix}add_vector_idx_{table_name}_{vector_field_name}",
                check_func=lambda: self._index_exists(table_name, vector_index_name),
                process_func=lambda: self._add_vector_index(table_name, vector_field_name),
            )

        with self._vector_column_cache_lock:
            self._vector_column_cache.add(cache_key)

    """
    Search SQL builders - PostgreSQL/Vastbase dialect
    """

    def _execute_search_sql(self, sql: str, params: tuple = None) -> tuple:
        """Execute a search SQL query and return results with timing.

        Args:
            sql: SQL query to execute.
            params: Optional query parameters.

        Returns:
            Tuple of (rows, elapsed_time).
        """
        start_time = time.time()
        rows = self._execute_sql(sql, params)
        elapsed_time = time.time() - start_time
        return rows, elapsed_time

    def _parse_fulltext_columns(
            self,
            fulltext_query: str,
            fulltext_columns: list[str]
    ) -> tuple:
        """Parse fulltext search columns with optional weight suffix and build search expressions.

        Args:
            fulltext_query: The escaped fulltext query string.
            fulltext_columns: List of column names, optionally with weight suffix (e.g., "col^0.5").

        Returns:
            Tuple of (fulltext_search_expr dict, fulltext_search_weight dict)
            where weights are normalized to 0~1.
        """
        fulltext_search_expr: dict[str, str] = {}
        fulltext_search_weight: dict[str, float] = {}

        for field in fulltext_columns:
            parts = field.split("^")
            column_name: str = parts[0]
            column_weight: float = float(parts[1]) if (len(parts) > 1 and parts[1]) else 1.0

            fulltext_search_weight[column_name] = column_weight
            # ADAPT: Vastbase BM25 expression replaces MySQL MATCH() AGAINST()
            fulltext_search_expr[column_name] = fulltext_search_template % (column_name, fulltext_query)

        weight_sum = sum(fulltext_search_weight.values())
        n = len(fulltext_search_weight)
        if weight_sum <= 0 < n:
            for column_name in fulltext_search_weight:
                fulltext_search_weight[column_name] = 1.0 / n
        else:
            for column_name in fulltext_search_weight:
                fulltext_search_weight[column_name] = fulltext_search_weight[column_name] / weight_sum

        return fulltext_search_expr, fulltext_search_weight

    def _build_vector_search_sql(
            self,
            table_name: str,
            fields_expr: str,
            vector_search_score_expr: str,
            filters_expr: str,
            vector_search_filter: str,
            vector_search_expr: str,
            limit: int,
            vector_topn: int,
            offset: int = 0
    ) -> str:
        # ADAPT: Standard LIMIT/OFFSET replaces APPROXIMATE LIMIT + OFFSET
        sql = (
            f"SELECT {fields_expr}, {vector_search_score_expr} AS _score"
            f"  FROM \"{table_name}\""
            f"  WHERE {filters_expr} AND {vector_search_filter}"
            f"  ORDER BY {vector_search_expr}"
            f"  LIMIT {limit if limit != 0 else vector_topn}"
        )
        if offset != 0:
            sql += f" OFFSET {offset}"
        return sql

    def _build_fulltext_search_sql(
            self,
            table_name: str,
            fields_expr: str,
            fulltext_search_score_expr: str,
            filters_expr: str,
            fulltext_search_filter: str,
            offset: int,
            limit: int,
            fulltext_topn: int,
            hint: str = ""
    ) -> str:
        # ADAPT: PostgreSQL LIMIT/OFFSET replaces MySQL LIMIT offset,limit
        effective_limit = limit if limit != 0 else fulltext_topn
        hint_expr = f"{hint} " if hint else ""
        return (
            f"SELECT {hint_expr}{fields_expr}, {fulltext_search_score_expr} AS _score"
            f"  FROM \"{table_name}\""
            f"  WHERE {filters_expr} AND {fulltext_search_filter}"
            f"  ORDER BY _score DESC"
            f"  LIMIT {effective_limit} OFFSET {offset}"
        )

    def _build_filter_search_sql(
            self,
            table_name: str,
            fields_expr: str,
            filters_expr: str,
            order_by_expr: str = "",
            limit_expr: str = ""
    ) -> str:
        return (
            f"SELECT {fields_expr}"
            f"  FROM \"{table_name}\""
            f"  WHERE {filters_expr}"
            f"  {order_by_expr} {limit_expr}"
        )

    def _build_count_sql(
            self,
            table_name: str,
            filters_expr: str,
            extra_filter: str = "",
            hint: str = ""
    ) -> str:
        hint_expr = f"{hint} " if hint else ""
        where_clause = f"{filters_expr} AND {extra_filter}" if extra_filter else filters_expr
        return f"SELECT {hint_expr}COUNT(id) FROM \"{table_name}\" WHERE {where_clause}"

    def _row_to_entity(self, data, fields: list[str]) -> dict:
        """Convert a database row to a dict entity.

        Args:
            data: Row data (tuple/list from cursor.fetchone()).
            fields: List of field names.

        Returns:
            Dict mapping field names to values.
        """
        entity = {}
        for i, field in enumerate(fields):
            value = data[i]
            if value is None:
                continue
            entity[field] = value
        return entity

    def _get_dataset_id_field(self) -> str:
        return "kb_id"

    def _get_filters(self, condition: dict) -> list[str]:
        """Convert condition dict to SQL filter expressions.

        Args:
            condition: Dict of filter conditions.

        Returns:
            List of SQL filter strings.
        """
        filters: list[str] = []
        for k, v in condition.items():
            if not v:
                continue
            if k == "exists":
                filters.append(f"\"{v}\" IS NOT NULL")
            elif k == "must_not" and isinstance(v, dict) and "exists" in v:
                filters.append(f"\"{v.get('exists')}\" IS NULL")
            elif isinstance(v, list):
                values: list[str] = []
                for item in v:
                    values.append(get_value_str(item))
                value = ", ".join(values)
                filters.append(f"\"{k}\" IN ({value})")
            else:
                filters.append(f"\"{k}\" = {get_value_str(v)}")
        return filters

    """
    CRUD operations
    """

    def get(self, doc_id: str, index_name: str, dataset_ids: list[str]) -> dict | None:
        """Get a single document by ID.

        Args:
            doc_id: Document ID to retrieve.
            index_name: Table name.
            dataset_ids: List of dataset IDs (unused in base implementation).

        Returns:
            Document dict or None if not found.
        """
        if not self._check_table_exists_cached(index_name):
            return None
        try:
            # ADAPT: Raw SQL SELECT replaces ObVecClient.get()
            cols = [col.name for col in self.get_column_definitions()]
            cols_str = ", ".join(f'"{c}"' for c in cols)
            rows = self._execute_sql(
                f"SELECT {cols_str} FROM \"{index_name}\" WHERE \"id\" = %s",
                (doc_id,),
            )
            if not rows:
                return None
            return self._row_to_entity(rows[0], fields=cols)
        except Exception as e:
            self.logger.exception(f"VastbaseConnectionBase.get({doc_id}) got exception")
            raise e

    def delete(self, condition: dict, index_name: str, dataset_id: str) -> int:
        """Delete rows matching the given condition.

        Args:
            condition: Filter condition dict.
            index_name: Table name.
            dataset_id: Dataset identifier.

        Returns:
            Number of rows deleted.
        """
        if not self._check_table_exists_cached(index_name):
            return 0
        if not index_name.startswith("ragflow_doc_meta_"):
            condition[self._get_dataset_id_field()] = dataset_id
        try:
            filters = self._get_filters(condition)
            where_clause = " AND ".join(filters) if filters else "1=1"

            # Get IDs first
            rows = self._execute_sql(
                f"SELECT \"id\" FROM \"{index_name}\" WHERE {where_clause}"
            )
            if not rows:
                return 0
            ids = [row[0] for row in rows]
            self.logger.debug(f"VastbaseConnectionBase.delete, filters: {condition}, ids: {ids}")

            # Delete by IDs
            placeholders = ", ".join(["%s"] * len(ids))
            self._execute_sql(
                f"DELETE FROM \"{index_name}\" WHERE \"id\" IN ({placeholders})",
                tuple(ids),
            )
            return len(ids)
        except Exception as e:
            self.logger.error(f"VastbaseConnectionBase.delete error: {str(e)}")
        return 0

    """
    Abstract CRUD methods that must be implemented by subclasses
    """

    @abstractmethod
    def search(
            self,
            select_fields: list[str],
            highlight_fields: list[str],
            condition: dict,
            match_expressions: list[MatchExpr],
            order_by: OrderByExpr,
            offset: int,
            limit: int,
            index_names: str | list[str],
            knowledgebase_ids: list[str],
            agg_fields: list[str] | None = None,
            rank_feature: dict | None = None,
            **kwargs,
    ):
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def insert(self, documents: list[dict], index_name: str, dataset_id: str = None) -> list[str]:
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def update(self, condition: dict, new_value: dict, index_name: str, dataset_id: str) -> bool:
        raise NotImplementedError("Not implemented")

    """
    Helper functions for search result - abstract methods
    """

    @abstractmethod
    def get_total(self, res) -> int:
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def get_doc_ids(self, res) -> list[str]:
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def get_fields(self, res, fields: list[str]) -> dict[str, dict]:
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def get_highlight(self, res, keywords: list[str], field_name: str):
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def get_aggregation(self, res, field_name: str):
        raise NotImplementedError("Not implemented")

    """
    SQL
    """

    def sql(self, sql: str, fetch_size: int, format: str):
        """Execute SQL query - default implementation.

        Args:
            sql: SQL query string.
            fetch_size: Maximum number of rows to fetch.
            format: Output format.

        Returns:
            Query results or None.
        """
        return None
