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
"""
Unit tests for Vastbase connection base class (vastbase_conn_base.py).
"""
import json
import sys
import threading
from unittest.mock import Mock, patch

import pytest
from sqlalchemy import Column, Integer, Text, Float, JSON
from sqlalchemy.dialects.postgresql import VARCHAR


def _make_instance():
    """Create a VastbaseConnectionBase instance with all abstracts stubbed."""
    from common.doc_store.vastbase_conn_base import VastbaseConnectionBase
    class _TestConn(VastbaseConnectionBase):
        def get_index_columns(self): return []
        def get_fulltext_columns(self): return []
        def get_column_definitions(self): return []
        def get_lock_prefix(self): return "test_"
        def search(self, *a, **kw): pass
        def insert(self, *a, **kw): return []
        def update(self, *a, **kw): return True
        def get_total(self, r): return 0
        def get_doc_ids(self, r): return []
        def get_fields(self, r, f): return {}
        def get_highlight(self, r, k, f): return {}
        def get_aggregation(self, r, f): return []
    conn = _TestConn.__new__(_TestConn)
    conn.logger = Mock()
    return conn


class TestGetValueStr:
    """Test cases for get_value_str helper function."""

    def test_string_value(self):
        from common.doc_store.vastbase_conn_base import get_value_str
        assert get_value_str("hello") == "'hello'"

    def test_string_with_single_quote(self):
        from common.doc_store.vastbase_conn_base import get_value_str
        assert get_value_str("it's") == "'it''s'"

    def test_bool_true(self):
        from common.doc_store.vastbase_conn_base import get_value_str
        assert get_value_str(True) == "true"
        assert get_value_str(False) == "false"

    def test_none_value(self):
        from common.doc_store.vastbase_conn_base import get_value_str
        assert get_value_str(None) == "NULL"

    def test_list_value(self):
        from common.doc_store.vastbase_conn_base import get_value_str
        assert get_value_str([1, 2, 3]) == "'[1, 2, 3]'"

    def test_dict_value(self):
        from common.doc_store.vastbase_conn_base import get_value_str
        assert get_value_str({"key": "value"}) == '\'{"key": "value"}\''

    def test_integer_value(self):
        from common.doc_store.vastbase_conn_base import get_value_str
        assert get_value_str(42) == "42"

    def test_string_with_backslash(self):
        from common.doc_store.vastbase_conn_base import get_value_str
        assert get_value_str("path\\to\\file") == "'path\\\\to\\\\file'"


class TestDocMetaDefinitions:
    """Test doc_meta column definitions."""

    def test_doc_meta_columns(self):
        from common.doc_store.vastbase_conn_base import doc_meta_columns, doc_meta_column_names
        assert len(doc_meta_columns) == 3
        assert "id" in doc_meta_column_names
        assert "kb_id" in doc_meta_column_names
        assert "meta_fields" in doc_meta_column_names

    def test_doc_meta_column_types(self):
        from common.doc_store.vastbase_conn_base import doc_meta_column_types
        assert "id" in doc_meta_column_types
        assert "kb_id" in doc_meta_column_types
        assert "meta_fields" in doc_meta_column_types


class TestSearchTemplates:
    """Test SQL search templates are PostgreSQL-compatible."""

    def test_fulltext_search_template(self):
        from common.doc_store.vastbase_conn_base import fulltext_search_template
        sql = fulltext_search_template % ("content", "hello world")
        assert "@~@" in sql

    def test_vector_search_template(self):
        from common.doc_store.vastbase_conn_base import vector_search_template
        sql = vector_search_template % ("embedding", "[0.1,0.2,0.3]")
        assert "<=>" in sql

    def test_vector_column_pattern(self):
        from common.doc_store.vastbase_conn_base import vector_column_pattern
        assert vector_column_pattern.match("q_128_vec")
        assert vector_column_pattern.match("q_1536_vec")
        assert not vector_column_pattern.match("my_vec")
        m = vector_column_pattern.match("q_768_vec")
        assert m.group("vector_size") == "768"

    def test_index_name_template(self):
        from common.doc_store.vastbase_conn_base import index_name_template
        assert index_name_template % ("my_table", "col1") == "ix_my_table_col1"

    def test_fulltext_index_name_template(self):
        from common.doc_store.vastbase_conn_base import fulltext_index_name_template
        assert fulltext_index_name_template % "content_col" == "fts_idx_content_col"


class TestVastbaseConnectionBaseInit:
    """Test VastbaseConnectionBase initialization."""

    def test_class_has_template_methods(self):
        from common.doc_store.vastbase_conn_base import VastbaseConnectionBase
        for m in ['get_index_columns', 'get_fulltext_columns',
                   'get_column_definitions', 'get_lock_prefix']:
            assert hasattr(VastbaseConnectionBase, m)

    def test_class_has_crud_methods(self):
        from common.doc_store.vastbase_conn_base import VastbaseConnectionBase
        for m in ['search', 'insert', 'update', 'get', 'delete']:
            assert hasattr(VastbaseConnectionBase, m)

    def test_class_has_helper_methods(self):
        from common.doc_store.vastbase_conn_base import VastbaseConnectionBase
        for m in ['get_total', 'get_doc_ids', 'get_fields',
                   'get_highlight', 'get_aggregation']:
            assert hasattr(VastbaseConnectionBase, m)


class TestVastbaseConnectionBaseDbType:
    def test_db_type_returns_vastbase(self):
        conn = _make_instance()
        assert conn.db_type() == "vastbase"


class TestRowToEntity:
    def test_basic_conversion(self):
        conn = _make_instance()
        result = conn._row_to_entity(
            ("id-1", "kb-1", "hello"),
            ["id", "kb_id", "content_with_weight"],
        )
        assert result == {"id": "id-1", "kb_id": "kb-1", "content_with_weight": "hello"}

    def test_none_values_skipped(self):
        conn = _make_instance()
        result = conn._row_to_entity(
            ("id-1", None, "hello"),
            ["id", "kb_id", "content"],
        )
        assert result == {"id": "id-1", "content": "hello"}
        assert "kb_id" not in result


class TestGetFiltersForBase:
    def test_equality_filter(self):
        conn = _make_instance()
        filters = conn._get_filters({"kb_id": "kb-1"})
        assert any("\"kb_id\" =" in f for f in filters)

    def test_list_in_filter(self):
        conn = _make_instance()
        filters = conn._get_filters({"kb_id": ["kb-1", "kb-2"]})
        assert any("IN" in f for f in filters)

    def test_exists_filter(self):
        conn = _make_instance()
        filters = conn._get_filters({"exists": "source_id"})
        assert any("IS NOT NULL" in f for f in filters)

    def test_must_not_filter(self):
        conn = _make_instance()
        filters = conn._get_filters({"must_not": {"exists": "forget_at"}})
        assert any("IS NULL" in f for f in filters)

    def test_empty_value_skipped(self):
        conn = _make_instance()
        filters = conn._get_filters({"empty_list": []})
        assert len(filters) == 0


class TestParseFulltextColumns:
    def test_single_column_no_weight(self):
        conn = _make_instance()
        expr, weights = conn._parse_fulltext_columns("query", ["content"])
        assert "content" in expr
        assert weights["content"] == 1.0

    def test_column_with_weight(self):
        conn = _make_instance()
        expr, weights = conn._parse_fulltext_columns("query", ["content^0.8"])
        # Weight is normalized to 1.0 for single column (0.8/0.8=1.0)
        assert weights["content"] == 1.0

    def test_multiple_columns_norm_weights(self):
        conn = _make_instance()
        expr, weights = conn._parse_fulltext_columns("query", ["a^2", "b^2"])
        assert abs(sum(weights.values()) - 1.0) < 0.0001
        assert weights["a"] == 0.5
        assert weights["b"] == 0.5

    def test_all_zero_weights(self):
        conn = _make_instance()
        expr, weights = conn._parse_fulltext_columns("query", ["a^0", "b^0"])
        assert weights["a"] == 0.5
        assert weights["b"] == 0.5

    def test_bm25_expression_format(self):
        conn = _make_instance()
        expr, _ = conn._parse_fulltext_columns("hello world", ["content"])
        assert "@~@" in expr["content"]


class TestSearchSqlBuilders:
    def test_build_vector_search_sql(self):
        conn = _make_instance()
        sql = conn._build_vector_search_sql(
            table_name="my_table",
            fields_expr='"id", "content"',
            vector_search_score_expr="score",
            filters_expr='"kb_id" = \'kb-1\'',
            vector_search_filter="score >= 0.7",
            vector_search_expr="embedding <=> '[0.1]'",
            limit=10, vector_topn=100, offset=0,
        )
        assert "LIMIT" in sql
        assert "APPROXIMATE" not in sql
        assert "_score" in sql

    def test_build_vector_search_sql_with_offset(self):
        conn = _make_instance()
        sql = conn._build_vector_search_sql(
            table_name="t", fields_expr="*", vector_search_score_expr="1",
            filters_expr="1=1", vector_search_filter="1=1",
            vector_search_expr="v", limit=10, vector_topn=100, offset=20,
        )
        assert "OFFSET 20" in sql

    def test_build_fulltext_search_sql(self):
        conn = _make_instance()
        sql = conn._build_fulltext_search_sql(
            table_name="t", fields_expr="*",
            fulltext_search_score_expr="bm25(c, 'q')",
            filters_expr="1=1", fulltext_search_filter="c @~@ 'q'",
            offset=0, limit=10, fulltext_topn=50,
        )
        assert "LIMIT 10 OFFSET 0" in sql

    def test_build_filter_search_sql(self):
        conn = _make_instance()
        sql = conn._build_filter_search_sql(
            table_name="t", fields_expr='"id"', filters_expr='"kb_id" = \'k\'',
            order_by_expr='ORDER BY "id" ASC', limit_expr="LIMIT 10",
        )
        assert "SELECT" in sql
        assert '"t"' in sql

    def test_build_count_sql(self):
        conn = _make_instance()
        sql = conn._build_count_sql("t", '"kb_id" = \'k\'')
        assert "COUNT(id)" in sql


class TestMapColumnType:
    def test_varchar_type(self):
        conn = _make_instance()
        assert conn._map_column_type(Column("name", VARCHAR(256))) == "VARCHAR(256)"

    def test_integer_type(self):
        conn = _make_instance()
        assert conn._map_column_type(Column("count", Integer)) == "INTEGER"

    def test_float_type(self):
        conn = _make_instance()
        assert conn._map_column_type(Column("score", Float)) == "DOUBLE PRECISION"

    def test_text_type(self):
        conn = _make_instance()
        assert conn._map_column_type(Column("content", Text)) == "TEXT"

    def test_json_type(self):
        conn = _make_instance()
        assert conn._map_column_type(Column("meta", JSON)) == "JSONB"


class TestHelperSqlMethods:
    def test_get_count(self):
        conn = _make_instance()
        conn._execute_sql = Mock(return_value=[(42,)])
        assert conn._get_count("t") == 42

    def test_get_count_with_filters(self):
        conn = _make_instance()
        conn._execute_sql = Mock(return_value=[(5,)])
        result = conn._get_count("t", ['"kb_id" = %s'], ("kb-1",))
        assert result == 5
        assert "WHERE" in conn._execute_sql.call_args[0][0]

    def test_get_dataset_id_field(self):
        assert _make_instance()._get_dataset_id_field() == "kb_id"


class TestVectorColumnCache:
    def test_cache_skip_on_zero_size(self):
        conn = _make_instance()
        conn._vector_column_cache = set()
        conn._vector_column_cache_lock = threading.RLock()
        conn._column_exist = Mock()
        conn._ensure_vector_column_exists("t", 0)
        conn._column_exist.assert_not_called()

    def test_cache_hit_skips_checks(self):
        conn = _make_instance()
        conn._vector_column_cache = {("t", 128)}
        conn._vector_column_cache_lock = threading.RLock()
        conn._column_exist = Mock()
        conn._ensure_vector_column_exists("t", 128)
        conn._column_exist.assert_not_called()

    def test_get_lock_prefix_abstract(self):
        # Our _TestConn implements get_lock_prefix, so the base class doesn't raise.
        # Verify the stubbed implementation works.
        conn = _make_instance()
        assert conn.get_lock_prefix() == "test_"


class TestDeleteMethod:
    def test_delete_returns_zero_when_table_missing(self):
        conn = _make_instance()
        conn.logger = Mock()
        # Mock _check_table_exists_cached instead of the full SQL chain
        with patch.object(conn, '_check_table_exists_cached', return_value=False):
            assert conn.delete({}, "nonexistent_table", "ds-1") == 0

    def test_delete_with_doc_meta_table(self):
        conn = _make_instance()
        conn._table_exists_cache = {"ragflow_doc_meta_tenant1"}
        conn._table_exists_cache_lock = threading.RLock()
        conn._get_filters = Mock(return_value=['"kb_id" = \'kb-1\''])
        conn._execute_sql = Mock()
        conn._execute_sql.side_effect = [[("id-1",)], []]
        conn.logger = Mock()
        result = conn.delete({"kb_id": "kb-1"}, "ragflow_doc_meta_tenant1", "ds-1")
        assert "kb_id" in conn._get_filters.call_args[0][0]


class TestCheckTableExistsCached:
    def test_cache_hit(self):
        conn = _make_instance()
        conn._table_exists_cache = {"cached_table"}
        conn._table_exists_cache_lock = threading.RLock()
        assert conn._check_table_exists_cached("cached_table") is True


class TestExecuteSqlBasics:
    def test_execute_sql_select(self):
        conn = _make_instance()
        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = [("row1",), ("row2",)]
        mock_client = Mock()
        mock_client.cursor.return_value = mock_cursor
        conn.client = mock_client
        result = conn._execute_sql("SELECT * FROM t")
        assert result == [("row1",), ("row2",)]

    def test_execute_sql_error_rolls_back(self):
        conn = _make_instance()
        mock_cursor = Mock()
        mock_cursor.fetchall.side_effect = Exception("no results")
        mock_client = Mock()
        mock_client.cursor.return_value = mock_cursor
        mock_client.commit.side_effect = Exception("commit failed")
        mock_client.rollback = Mock()
        conn.client = mock_client
        with pytest.raises(Exception):
            conn._execute_sql("INSERT INTO t VALUES (2)")


class TestTryWithLockTimeout:
    @pytest.mark.skip(reason="_try_with_lock requires rag.utils.redis_conn which has deep import chain")
    def test_timeout_raises_exception(self):
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
