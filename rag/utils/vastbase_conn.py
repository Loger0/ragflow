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
import re
import time
from typing import Any, Optional

import numpy as np
from pydantic import BaseModel
from sqlalchemy import Column, Integer, Text, Float

from common.constants import PAGERANK_FLD, TAG_FLD
from common.decorator import singleton
from common.doc_store.doc_store_base import (
    MatchExpr, OrderByExpr, FusionExpr,
    MatchTextExpr, MatchDenseExpr,
)
from common.doc_store.vastbase_conn_base import (
    VastbaseConnectionBase, get_value_str,
    vector_search_template, vector_column_pattern,
    fulltext_index_name_template, doc_meta_column_names,
    doc_meta_column_types,
)
from common.float_utils import get_float
from rag.nlp import rag_tokenizer

logger = logging.getLogger('ragflow.vastbase_conn')

# ADAPT: PostgreSQL Text type replaces MySQL LONGTEXT; JSON replaces JSON
# ADAPT: ARRAY types use PostgreSQL native array syntax

column_order_id = Column("_order_id", Integer, nullable=True, comment="chunk order id for maintaining sequence")
column_group_id = Column("group_id", Text, nullable=True, comment="group id for external retrieval")
column_mom_id = Column("mom_id", Text, nullable=True, comment="parent chunk id")
column_chunk_data = Column("chunk_data", Text, nullable=True, comment="table parser row data")
column_raptor_kwd = Column("raptor_kwd", Text, nullable=True, comment="RAPTOR summary marker")
column_raptor_layer_int = Column("raptor_layer_int", Integer, nullable=True, comment="RAPTOR summary layer")
column_n_hop_with_weight = Column("n_hop_with_weight", Text, nullable=True,
                                   comment="JSON-encoded n-hop neighbour paths and weights for a graph entity")

# ADAPT: Column definitions with PostgreSQL-compatible types
column_definitions: list[Column] = [
    Column("id", Text, primary_key=True, comment="chunk id"),
    Column("kb_id", Text, nullable=False, comment="knowledge base id"),
    Column("doc_id", Text, nullable=True, comment="document id"),
    Column("docnm_kwd", Text, nullable=True, comment="document name"),
    Column("doc_type_kwd", Text, nullable=True, comment="document type"),
    Column("title_tks", Text, nullable=True, comment="title tokens"),
    Column("title_sm_tks", Text, nullable=True, comment="fine-grained (small) title tokens"),
    Column("content_with_weight", Text, nullable=True, comment="the original content"),
    Column("content_ltks", Text, nullable=True, comment="long text tokens derived from content_with_weight"),
    Column("content_sm_ltks", Text, nullable=True, comment="fine-grained (small) tokens derived from content_ltks"),
    Column("pagerank_fea", Integer, nullable=True, comment="page rank priority, usually set in kb level"),
    Column("important_kwd", Text, nullable=True, comment="keywords stored as JSON array"),
    Column("important_tks", Text, nullable=True, comment="keyword tokens"),
    Column("question_kwd", Text, nullable=True, comment="questions stored as JSON array"),
    Column("question_tks", Text, nullable=True, comment="question tokens"),
    Column("tag_kwd", Text, nullable=True, comment="tags stored as JSON array"),
    Column("tag_feas", Text, nullable=True,
           comment="tag features used for 'rank_feature', format: JSON [tag -> relevance score]"),
    Column("available_int", Integer, nullable=False, default=1, comment="status: 0 for unavailable, 1 for available"),
    Column("create_time", Text, nullable=True, comment="creation time in YYYY-MM-DD HH:MM:SS format"),
    Column("create_timestamp_flt", Float, nullable=True, comment="creation timestamp in float format"),
    Column("img_id", Text, nullable=True, comment="image id"),
    Column("position_int", Text, nullable=True, comment="position stored as JSON array"),
    Column("page_num_int", Text, nullable=True, comment="page number stored as JSON array"),
    Column("top_int", Text, nullable=True, comment="rank from top stored as JSON array"),
    Column("knowledge_graph_kwd", Text, nullable=True, comment="knowledge graph chunk type"),
    Column("source_id", Text, nullable=True, comment="source document id stored as JSON array"),
    Column("entity_kwd", Text, nullable=True, comment="entity name"),
    Column("entity_type_kwd", Text, nullable=True, comment="entity type"),
    Column("from_entity_kwd", Text, nullable=True, comment="the source entity of this edge"),
    Column("to_entity_kwd", Text, nullable=True, comment="the target entity of this edge"),
    Column("weight_int", Integer, nullable=True, comment="the weight of this edge"),
    Column("weight_flt", Float, nullable=True, comment="the weight of community report"),
    Column("entities_kwd", Text, nullable=True, comment="node ids of entities stored as JSON array"),
    Column("rank_flt", Float, nullable=True, comment="rank of this entity"),
    column_n_hop_with_weight,
    Column("removed_kwd", Text, nullable=True, default="N", comment="whether it has been deleted"),
    column_raptor_kwd,
    column_raptor_layer_int,
    column_chunk_data,
    Column("metadata", Text, nullable=True, comment="metadata for this chunk stored as JSON"),
    Column("extra", Text, nullable=True, comment="extra information of non-general chunk stored as JSON"),
    column_order_id,
    column_group_id,
    column_mom_id,
]

column_names: list[str] = [col.name for col in column_definitions]
column_types: dict = {col.name: col.type for col in column_definitions}

# ADAPT: Array-like columns stored as JSON Text in Vastbase (PostgreSQL supports native arrays,
# but for simplicity we store as TEXT and handle JSON serialization in application code)
array_columns: list[str] = [
    "important_kwd", "question_kwd", "tag_kwd",
    "position_int", "page_num_int", "top_int",
    "source_id", "entities_kwd",
]

INDEX_COLUMNS: list[str] = [
    "kb_id",
    "doc_id",
    "available_int",
    "knowledge_graph_kwd",
    "entity_type_kwd",
    "removed_kwd",
]

FTS_COLUMNS_ORIGIN: list[str] = [
    "docnm_kwd^10",
    "content_with_weight",
    "important_tks^20",
    "question_tks^20",
]

FTS_COLUMNS_TKS: list[str] = [
    "title_tks^10",
    "title_sm_tks^5",
    "important_tks^20",
    "question_tks^20",
    "content_ltks^2",
    "content_sm_ltks",
]

EXTRA_COLUMNS: list[Column] = [
    column_order_id,
    column_group_id,
    column_mom_id,
    column_chunk_data,
    column_raptor_kwd,
    column_raptor_layer_int,
    column_n_hop_with_weight,
]


class SearchResult(BaseModel):
    total: int
    chunks: list[dict]


def get_column_value(column_name: str, value: Any) -> Any:
    """Convert raw column value to its Python type based on column definition.

    Args:
        column_name: Name of the column.
        value: Raw value from the database.

    Returns:
        Converted Python value.
    """
    column_type = column_types.get(column_name) or doc_meta_column_types.get(column_name)
    if column_type:
        if isinstance(column_type, (Text,)):
            return str(value)
        elif isinstance(column_type, Integer):
            return int(value)
        elif isinstance(column_type, Float):
            return float(value)
        elif column_name in array_columns:
            if isinstance(value, str):
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    return value
            else:
                return value
        else:
            return value
    elif vector_column_pattern.match(column_name):
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
        else:
            return value
    elif column_name == "_score":
        return float(value)
    else:
        raise ValueError(f"Unknown column '{column_name}' with value '{value}'.")


def get_default_value(column_name: str) -> Any:
    """Get default value for a column.

    Args:
        column_name: Name of the column.

    Returns:
        Default value.
    """
    if column_name == "available_int":
        return 1
    elif column_name == "removed_kwd":
        return "N"
    elif column_name == "_order_id":
        return 0
    else:
        return None


# ADAPT: PostgreSQL JSONB operators replace MySQL JSON_EXTRACT/CONTAINS
def get_metadata_filter_expression(metadata_filtering_conditions: dict) -> str:
    """Convert metadata filtering conditions to PostgreSQL JSONB expression.

    Args:
        metadata_filtering_conditions: dict with 'conditions' and 'logical_operator' keys.

    Returns:
        PostgreSQL JSONB path expression string.
    """
    if not metadata_filtering_conditions:
        return ""

    conditions = metadata_filtering_conditions.get("conditions", [])
    logical_operator = metadata_filtering_conditions.get("logical_operator", "and").upper()

    if not conditions:
        return ""

    if logical_operator not in ["AND", "OR"]:
        raise ValueError(f"Unsupported logical operator: {logical_operator}. Only 'and' and 'or' are supported.")

    metadata_filters = []
    for condition in conditions:
        name = condition.get("name")
        comparison_operator = condition.get("comparison_operator")
        value = condition.get("value")

        if not all([name, comparison_operator]):
            continue

        # ADAPT: PostgreSQL JSONB ->> operator replaces MySQL JSON_EXTRACT
        expr = f"metadata->>'{name}'"
        value_str = get_value_str(value)

        if comparison_operator == "is":
            metadata_filters.append(f"{expr} = {value_str}")
        elif comparison_operator == "is not":
            metadata_filters.append(f"({expr} != {value_str} OR {expr} IS NULL)")
        elif comparison_operator == "contains":
            metadata_filters.append(f"{expr} LIKE '%' || {value_str} || '%'")
        elif comparison_operator == "not contains":
            metadata_filters.append(f"{expr} NOT LIKE '%' || {value_str} || '%'")
        elif comparison_operator == "start with":
            metadata_filters.append(f"{expr} LIKE {value_str} || '%'")
        elif comparison_operator == "end with":
            metadata_filters.append(f"{expr} LIKE '%' || {value_str}")
        elif comparison_operator == "empty":
            metadata_filters.append(
                f"({expr} IS NULL OR {expr} = '' OR {expr} = '[]' OR {expr} = '{{}}')"
            )
        elif comparison_operator == "not empty":
            metadata_filters.append(
                f"({expr} IS NOT NULL AND {expr} != '' AND {expr} != '[]' AND {expr} != '{{}}')"
            )
        elif comparison_operator == "=":
            metadata_filters.append(f"({expr})::numeric = {value_str}")
        elif comparison_operator == "≠":
            metadata_filters.append(f"({expr})::numeric != {value_str}")
        elif comparison_operator == ">":
            metadata_filters.append(f"({expr})::numeric > {value_str}")
        elif comparison_operator == "<":
            metadata_filters.append(f"({expr})::numeric < {value_str}")
        elif comparison_operator == "≥":
            metadata_filters.append(f"({expr})::numeric >= {value_str}")
        elif comparison_operator == "≤":
            metadata_filters.append(f"({expr})::numeric <= {value_str}")
        elif comparison_operator == "before":
            metadata_filters.append(f"({expr})::timestamp < {value_str}::timestamp")
        elif comparison_operator == "after":
            metadata_filters.append(f"({expr})::timestamp > {value_str}::timestamp")
        else:
            logger.warning(f"Unsupported comparison operator: {comparison_operator}")
            continue

    if not metadata_filters:
        return ""

    return f"({f' {logical_operator} '.join(metadata_filters)})"


_VALID_FILTER_COLUMNS: set[str] = set(column_names) | set(doc_meta_column_names)


def get_filters(condition: dict) -> list[str]:
    """Convert condition dict to PostgreSQL SQL filter expressions.

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
            if isinstance(v, str) and v in _VALID_FILTER_COLUMNS:
                filters.append(f"\"{v}\" IS NOT NULL")
        elif k == "must_not" and isinstance(v, dict) and "exists" in v:
            col = v.get("exists")
            if isinstance(col, str) and col in _VALID_FILTER_COLUMNS:
                filters.append(f"\"{col}\" IS NULL")
        elif k == "metadata_filtering_conditions":
            metadata_filter = get_metadata_filter_expression(v)
            if metadata_filter:
                filters.append(metadata_filter)
        elif k in array_columns:
            if isinstance(v, list):
                array_filters = []
                for vv in v:
                    array_filters.append(
                        f"\"{k}\"::jsonb @> {get_value_str(json.dumps([vv], ensure_ascii=False))}::jsonb"
                    )
                array_filter = " OR ".join(array_filters)
                filters.append(f"({array_filter})")
            else:
                filters.append(
                    f"\"{k}\"::jsonb @> {get_value_str(json.dumps([v], ensure_ascii=False))}::jsonb"
                )
        elif k in _VALID_FILTER_COLUMNS:
            if isinstance(v, list):
                values: list[str] = []
                for item in v:
                    values.append(get_value_str(item))
                value = ", ".join(values)
                filters.append(f"\"{k}\" IN ({value})")
            else:
                filters.append(f"\"{k}\" = {get_value_str(v)}")
    return filters


@singleton
class VBConnection(VastbaseConnectionBase):
    """Vastbase document store connection for RAG chunk storage.

    Inherits from VastbaseConnectionBase. Implements search (fusion/vector/fulltext/
    aggregation/filter 5 paths), insert, update, and all abstract getter methods.
    """

    def __init__(self):
        super().__init__(logger_name='ragflow.vastbase_conn')
        self._fulltext_search_columns = FTS_COLUMNS_ORIGIN if self.search_original_content else FTS_COLUMNS_TKS

    """
    Template method implementations
    """

    def get_index_columns(self) -> list[str]:
        return INDEX_COLUMNS

    def get_column_definitions(self) -> list[Column]:
        return column_definitions

    def get_extra_columns(self) -> list[Column]:
        return EXTRA_COLUMNS

    def get_lock_prefix(self) -> str:
        return "vb_"

    def _get_filters(self, condition: dict) -> list[str]:
        return get_filters(condition)

    def get_fulltext_columns(self) -> list[str]:
        """Return list of column names that need fulltext indexes (without weight suffix)."""
        return [col.split("^")[0] for col in self._fulltext_search_columns]

    def delete_idx(self, index_name: str, dataset_id: str):
        if dataset_id:
            return
        super().delete_idx(index_name, dataset_id)

    """
    CRUD operations
    """

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
        if agg_fields is None:
            agg_fields = []

        if isinstance(index_names, str):
            index_names = index_names.split(",")
        if not (isinstance(index_names, list) and len(index_names) > 0):
            raise ValueError("index_names must be a non-empty list")
        index_names = list(set(index_names))

        # Filter out fulltext search when disabled
        if len(match_expressions) == 3:
            if not self.enable_fulltext_search:
                match_expressions = [m for m in match_expressions if isinstance(m, MatchDenseExpr)]
            else:
                for m in match_expressions:
                    if isinstance(m, FusionExpr):
                        weights = m.fusion_params["weights"]
                        vector_similarity_weight = get_float(weights.split(",")[1])
                        if vector_similarity_weight <= 0.0:
                            match_expressions = [m for m in match_expressions if isinstance(m, MatchTextExpr)]
                        elif vector_similarity_weight >= 1.0:
                            match_expressions = [m for m in match_expressions if isinstance(m, MatchDenseExpr)]

        result: SearchResult = SearchResult(
            total=0,
            chunks=[],
        )

        output_fields = select_fields.copy()
        if "*" in output_fields:
            if index_names[0].startswith("ragflow_doc_meta_"):
                output_fields = doc_meta_column_names.copy()
            else:
                output_fields = column_names.copy()

        if "id" not in output_fields:
            output_fields = ["id"] + output_fields
        if "_score" in output_fields:
            output_fields.remove("_score")

        if highlight_fields:
            for field in highlight_fields:
                if field not in output_fields:
                    output_fields.append(field)

        fields_expr = ", ".join([f'"{f}"' for f in output_fields])

        condition["kb_id"] = knowledgebase_ids
        filters: list[str] = get_filters(condition)
        filters_expr = " AND ".join(filters)

        fulltext_query: Optional[str] = None
        fulltext_topn: Optional[int] = None
        fulltext_search_weight: dict[str, float] = {}
        fulltext_search_expr: dict[str, str] = {}
        fulltext_search_idx_list: list[str] = []
        fulltext_search_score_expr: Optional[str] = None
        fulltext_search_filter: Optional[str] = None

        vector_column_name: Optional[str] = None
        vector_data: Optional[list[float]] = None
        vector_topn: Optional[int] = None
        vector_similarity_threshold: Optional[float] = None
        vector_similarity_weight: Optional[float] = None
        vector_search_expr: Optional[str] = None
        vector_search_score_expr: Optional[str] = None
        vector_search_filter: Optional[str] = None

        for m in match_expressions:
            if isinstance(m, MatchTextExpr):
                if "original_query" not in m.extra_options:
                    raise ValueError("'original_query' is missing in extra_options.")
                fulltext_query = m.extra_options["original_query"]
                fulltext_query = fulltext_query.strip()
                fulltext_topn = m.topn

                fulltext_search_expr, fulltext_search_weight = self._parse_fulltext_columns(
                    fulltext_query, self._fulltext_search_columns
                )
                for column_name in fulltext_search_expr.keys():
                    fulltext_search_idx_list.append(fulltext_index_name_template % column_name)

            elif isinstance(m, MatchDenseExpr):
                if m.embedding_data_type != "float":
                    raise ValueError(f"embedding data type '{m.embedding_data_type}' is not float.")
                vector_column_name = m.vector_column_name
                vector_data = m.embedding_data
                vector_topn = m.topn
                vector_similarity_threshold = float(m.extra_options.get("similarity", 0.0))
            elif isinstance(m, FusionExpr):
                weights = m.fusion_params["weights"]
                vector_similarity_weight = get_float(weights.split(",")[1])

        if fulltext_query:
            # ADAPT: Vastbase BM25 @~@ operator
            query_escaped = fulltext_query.replace("'", "''")
            fulltext_search_filter = (
                f"({' OR '.join([f'\"{col}\" @~@ ' + get_value_str(query_escaped) for col in fulltext_search_expr.keys()])})"
            )
            fulltext_search_score_expr = (
                f"({' + '.join(f'bm25(\"{col}\", ' + get_value_str(query_escaped) + f') * {fulltext_search_weight.get(col, 0)}' for col in fulltext_search_expr.keys())})"
            )

        if vector_data:
            vector_data_str = "[" + ",".join([str(np.float32(v)) for v in vector_data]) + "]"
            # ADAPT: pgvector <=> operator replaces cosine_distance()
            vector_search_expr = vector_search_template % (vector_column_name, vector_data_str)
            # ADAPT: (1 - <=>) gives cosine similarity score [-1, 1]
            vector_search_score_expr = f"(1 - ({vector_search_expr}))"
            vector_search_filter = f"{vector_search_score_expr} >= {vector_similarity_threshold}"

        pagerank_score_expr = f"(COALESCE(CAST(\"{PAGERANK_FLD}\" AS FLOAT), 0) / 100)"

        if fulltext_query and vector_data:
            search_type = "fusion"
        elif fulltext_query:
            search_type = "fulltext"
        elif vector_data:
            search_type = "vector"
        elif len(agg_fields) > 0:
            search_type = "aggregation"
        else:
            search_type = "filter"

        if search_type in ["fusion", "fulltext", "vector"] and "_score" not in output_fields:
            output_fields.append("_score")

        # Recompute fields_expr with updated output_fields
        fields_expr = ", ".join([f'"{f}"' for f in output_fields])

        if limit:
            if vector_topn is not None:
                limit = min(vector_topn, limit)
            if fulltext_topn is not None:
                limit = min(fulltext_topn, limit)

        for index_name in index_names:
            if not self._check_table_exists_cached(index_name):
                continue

            if search_type == "fusion":
                # ADAPT: PostgreSQL CTE with standard LIMIT/OFFSET replaces APPROXIMATE LIMIT
                num_candidates = (vector_topn or limit) + (fulltext_topn or limit)

                score_expr = (
                    f"(fts.relevance * {1 - vector_similarity_weight} + "
                    f"vs.similarity * {vector_similarity_weight} + {pagerank_score_expr})"
                )
                count_sql = (
                    f"WITH fulltext_results AS ("
                    f"  SELECT id FROM \"{index_name}\""
                    f"      WHERE {filters_expr} AND {fulltext_search_filter}"
                    f"      ORDER BY {fulltext_search_score_expr} DESC"
                    f"      LIMIT {fulltext_topn}"
                    f"),"
                    f"vector_results AS ("
                    f"  SELECT id FROM \"{index_name}\""
                    f"      WHERE {filters_expr} AND {vector_search_filter}"
                    f"      ORDER BY {vector_search_expr}"
                    f"      LIMIT {vector_topn}"
                    f")"
                    f"  SELECT COUNT(*) FROM fulltext_results f FULL OUTER JOIN vector_results v ON f.id = v.id"
                )
                logger.debug("VBConnection.search with count sql: %s", count_sql)
                rows, elapsed_time = self._execute_search_sql(count_sql)
                total_count = rows[0][0] if rows else 0
                result.total += total_count
                logger.info(
                    f"VBConnection.search table {index_name}, search type: fusion, step: 1-count, "
                    f"elapsed time: {elapsed_time:.3f}s, got count: {total_count}"
                )

                if total_count == 0:
                    continue

                fusion_sql = (
                    f"WITH fulltext_results AS ("
                    f"  SELECT f.id, f.pagerank_fea, {fulltext_search_score_expr} AS relevance"
                    f"      FROM \"{index_name}\" f"
                    f"      WHERE {filters_expr} AND {fulltext_search_filter}"
                    f"      ORDER BY relevance DESC"
                    f"      LIMIT {fulltext_topn}"
                    f"),"
                    f"vector_results AS ("
                    f"  SELECT v.id, v.pagerank_fea, {vector_search_score_expr} AS similarity"
                    f"      FROM \"{index_name}\" v"
                    f"      WHERE {filters_expr} AND {vector_search_filter}"
                    f"      ORDER BY {vector_search_expr}"
                    f"      LIMIT {vector_topn}"
                    f"),"
                    f"combined_results AS ("
                    f"  SELECT COALESCE(f.id, v.id) AS id,"
                    f"    COALESCE(f.pagerank_fea, v.pagerank_fea) AS pagerank_fea,"
                    f"    COALESCE(f.relevance, 0) AS relevance,"
                    f"    COALESCE(v.similarity, 0) AS similarity"
                    f"      FROM fulltext_results f"
                    f"      FULL OUTER JOIN vector_results v"
                    f"      ON f.id = v.id"
                    f")"
                    f"  SELECT {fields_expr}, ({score_expr}) AS _score"
                    f"      FROM combined_results c"
                    f"      JOIN \"{index_name}\" t ON c.id = t.id"
                    f"      ORDER BY _score DESC"
                    f"      LIMIT {limit} OFFSET {offset}"
                )
                logger.debug("VBConnection.search with fusion sql: %s", fusion_sql)
                rows, elapsed_time = self._execute_search_sql(fusion_sql)
                logger.info(
                    f"VBConnection.search table {index_name}, search type: fusion, step: 2-query, "
                    f"elapsed time: {elapsed_time:.3f}s, return rows: {len(rows)}"
                )

                for row in rows:
                    result.chunks.append(self._row_to_entity(row, output_fields))

            elif search_type == "vector":
                count_sql = self._build_count_sql(index_name, filters_expr, vector_search_filter)
                logger.debug("VBConnection.search with vector count sql: %s", count_sql)
                rows, elapsed_time = self._execute_search_sql(count_sql)
                total_count = rows[0][0] if rows else 0
                result.total += total_count
                logger.info(
                    f"VBConnection.search table {index_name}, search type: vector, step: 1-count, "
                    f"elapsed time: {elapsed_time:.3f}s, got count: {total_count}"
                )

                if total_count == 0:
                    continue

                vector_sql = self._build_vector_search_sql(
                    index_name, fields_expr, vector_search_score_expr, filters_expr,
                    vector_search_filter, vector_search_expr, limit, vector_topn, offset
                )
                logger.debug("VBConnection.search with vector sql: %s", vector_sql)
                rows, elapsed_time = self._execute_search_sql(vector_sql)
                logger.info(
                    f"VBConnection.search table {index_name}, search type: vector, step: 2-query, "
                    f"elapsed time: {elapsed_time:.3f}s, return rows: {len(rows)}"
                )

                for row in rows:
                    result.chunks.append(self._row_to_entity(row, output_fields))

            elif search_type == "fulltext":
                count_sql = self._build_count_sql(index_name, filters_expr, fulltext_search_filter)
                logger.debug("VBConnection.search with fulltext count sql: %s", count_sql)
                rows, elapsed_time = self._execute_search_sql(count_sql)
                total_count = rows[0][0] if rows else 0
                result.total += total_count
                logger.info(
                    f"VBConnection.search table {index_name}, search type: fulltext, step: 1-count, "
                    f"elapsed time: {elapsed_time:.3f}s, got count: {total_count}"
                )

                if total_count == 0:
                    continue

                fulltext_sql = self._build_fulltext_search_sql(
                    index_name, fields_expr, fulltext_search_score_expr, filters_expr,
                    fulltext_search_filter, offset, limit, fulltext_topn
                )
                logger.debug("VBConnection.search with fulltext sql: %s", fulltext_sql)
                rows, elapsed_time = self._execute_search_sql(fulltext_sql)
                logger.info(
                    f"VBConnection.search table {index_name}, search type: fulltext, step: 2-query, "
                    f"elapsed time: {elapsed_time:.3f}s, return rows: {len(rows)}"
                )

                for row in rows:
                    result.chunks.append(self._row_to_entity(row, output_fields))

            elif search_type == "aggregation":
                if len(agg_fields) != 1:
                    raise ValueError("Only one aggregation field is supported.")
                agg_field = agg_fields[0]
                if agg_field in array_columns:
                    rows = self._execute_sql(
                        f"SELECT \"{agg_field}\" FROM \"{index_name}\""
                        f" WHERE \"{agg_field}\" IS NOT NULL AND {filters_expr}"
                    )
                    counts = {}
                    for row in rows:
                        if row[0]:
                            val = row[0]
                            if isinstance(val, str):
                                try:
                                    arr = json.loads(val)
                                except json.JSONDecodeError:
                                    logger.warning(f"Failed to parse JSON array: {val}")
                                    continue
                            else:
                                arr = val

                            if isinstance(arr, list):
                                for vv in arr:
                                    if isinstance(vv, str) and vv.strip():
                                        counts[vv] = counts.get(vv, 0) + 1

                    for vv, count in counts.items():
                        result.chunks.append({
                            "value": vv,
                            "count": count,
                        })
                    result.total += len(counts)
                else:
                    rows = self._execute_sql(
                        f"SELECT \"{agg_field}\", COUNT(*) as cnt FROM \"{index_name}\""
                        f" WHERE \"{agg_field}\" IS NOT NULL AND {filters_expr}"
                        f" GROUP BY \"{agg_field}\""
                    )
                    for row in rows:
                        result.chunks.append({
                            "value": row[0],
                            "count": int(row[1]),
                        })
                        result.total += 1
            else:
                # filter search
                orders: list[str] = []
                if order_by:
                    for field, order in order_by.fields:
                        order_str = "ASC" if order == 0 else "DESC"
                        orders.append(f"\"{field}\" {order_str}")
                count_sql = self._build_count_sql(index_name, filters_expr)
                logger.debug("VBConnection.search with filter count sql: %s", count_sql)
                rows, elapsed_time = self._execute_search_sql(count_sql)
                total_count = rows[0][0] if rows else 0
                result.total += total_count
                logger.info(
                    f"VBConnection.search table {index_name}, search type: filter, step: 1-count, "
                    f"elapsed time: {elapsed_time:.3f}s, got count: {total_count}"
                )

                if total_count == 0:
                    continue

                order_by_expr = ("ORDER BY " + ", ".join(orders)) if len(orders) > 0 else ""
                # ADAPT: PostgreSQL LIMIT/OFFSET
                limit_expr = f"LIMIT {limit} OFFSET {offset}" if limit != 0 else ""
                filter_sql = self._build_filter_search_sql(
                    index_name, fields_expr, filters_expr, order_by_expr, limit_expr
                )
                logger.debug("VBConnection.search with filter sql: %s", filter_sql)
                rows, elapsed_time = self._execute_search_sql(filter_sql)
                logger.info(
                    f"VBConnection.search table {index_name}, search type: filter, step: 2-query, "
                    f"elapsed time: {elapsed_time:.3f}s, return rows: {len(rows)}"
                )

                for row in rows:
                    result.chunks.append(self._row_to_entity(row, output_fields))

        if result.total == 0:
            result.total = len(result.chunks)

        return result

    def get(self, chunk_id: str, index_name: str, knowledgebase_ids: list[str]) -> dict | None:
        try:
            doc = super().get(chunk_id, index_name, knowledgebase_ids)
            if doc is None:
                return None
            return doc
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error when getting chunk {chunk_id}: {str(e)}")
            return {
                "id": chunk_id,
                "error": f"Failed to parse chunk data due to invalid JSON: {str(e)}"
            }
        except Exception as e:
            logger.exception(f"VBConnection.get({chunk_id}) got exception")
            raise e

    def insert(self, documents: list[dict], index_name: str, knowledgebase_id: str = None) -> list[str]:
        """Insert documents into Vastbase table.

        Uses INSERT ... ON CONFLICT DO UPDATE (PostgreSQL upsert).

        Args:
            documents: List of document dicts to insert.
            index_name: Target table name.
            knowledgebase_id: Knowledge base identifier.

        Returns:
            List of error messages (empty if successful).
        """
        if not documents:
            return []

        if index_name.startswith("ragflow_doc_meta_"):
            return self._insert_doc_meta(documents, index_name)

        docs: list[dict] = []
        ids: list[str] = []
        for document in documents:
            d: dict = {}
            for k, v in document.items():
                if vector_column_pattern.match(k):
                    d[k] = v
                    continue
                if k not in column_names:
                    if "extra" not in d:
                        d["extra"] = {}
                    d["extra"][k] = v
                    continue
                if v is None:
                    d[k] = get_default_value(k)
                    continue

                if k == "kb_id" and isinstance(v, list):
                    d[k] = v[0]
                elif k == "content_with_weight" and isinstance(v, dict):
                    d[k] = json.dumps(v, ensure_ascii=False)
                elif k in array_columns:
                    if isinstance(v, list):
                        cleaned_v = []
                        for vv in v:
                            if isinstance(vv, str):
                                cleaned_str = vv.strip()
                                cleaned_str = cleaned_str.replace('\\', '\\\\').replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
                                cleaned_v.append(cleaned_str)
                            else:
                                cleaned_v.append(vv)
                        d[k] = json.dumps(cleaned_v, ensure_ascii=False)
                    else:
                        d[k] = json.dumps(v, ensure_ascii=False)
                else:
                    d[k] = v

            ids.append(d["id"])
            for column_name in column_names:
                if column_name not in d:
                    d[column_name] = get_default_value(column_name)

            metadata = d.get("metadata", {})
            if metadata is None:
                metadata = {}
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except json.JSONDecodeError:
                    metadata = {}
            group_id = metadata.get("_group_id")
            title = metadata.get("_title")
            if d.get("doc_id"):
                if group_id:
                    d["group_id"] = group_id
                else:
                    d["group_id"] = d["doc_id"]
                if title:
                    d["docnm_kwd"] = title

            # ADAPT: Convert extra dict to JSON string
            if "extra" in d and isinstance(d["extra"], dict):
                d["extra"] = json.dumps(d["extra"], ensure_ascii=False)
            if "metadata" in d and isinstance(d["metadata"], dict):
                d["metadata"] = json.dumps(d["metadata"], ensure_ascii=False)

            docs.append(d)

        logger.debug("VBConnection.insert chunks: ids=%s", ids)

        res = []
        try:
            self._batch_upsert(index_name, docs)
        except Exception as e:
            logger.error(f"VBConnection.insert error: {str(e)}")
            res.append(str(e))
        return res

    def _batch_upsert(self, table_name: str, docs: list[dict]):
        """Batch upsert documents into Vastbase using INSERT ON CONFLICT.

        Args:
            table_name: Target table name.
            docs: List of document dicts.
        """
        if not docs:
            return

        columns = list(docs[0].keys())
        col_names = ", ".join(f'"{c}"' for c in columns)

        row_values = []
        all_params = []
        for doc in docs:
            placeholders = []
            for c in columns:
                v = doc.get(c)
                if v is not None:
                    placeholders.append("%s")
                    all_params.append(v)
                else:
                    placeholders.append("%s")
                    all_params.append(None)
            row_values.append(f"({', '.join(placeholders)})")

        values_str = ", ".join(row_values)

        # ADAPT: PostgreSQL INSERT ON CONFLICT replaces ObVecClient.upsert()
        update_set = ", ".join([f'"{c}" = EXCLUDED."{c}"' for c in columns if c != "id"])

        sql = (
            f"INSERT INTO \"{table_name}\" ({col_names})"
            f" VALUES {values_str}"
            f" ON CONFLICT (id) DO UPDATE SET {update_set}"
        )

        self._execute_sql(sql, tuple(all_params))

    def _insert_doc_meta(self, documents: list[dict], index_name: str) -> list[str]:
        """Insert documents into doc_meta table.

        Args:
            documents: List of document dicts.
            index_name: Target table name.

        Returns:
            List of error messages.
        """
        docs: list[dict] = []
        for document in documents:
            d = {
                "id": document.get("id"),
                "kb_id": document.get("kb_id"),
            }
            meta_fields = document.get("meta_fields")
            if meta_fields is not None:
                if isinstance(meta_fields, dict):
                    d["meta_fields"] = json.dumps(meta_fields, ensure_ascii=False)
                elif isinstance(meta_fields, str):
                    d["meta_fields"] = meta_fields
                else:
                    d["meta_fields"] = "{}"
            else:
                d["meta_fields"] = "{}"
            docs.append(d)

        logger.debug("VBConnection._insert_doc_meta: %s", docs)

        res = []
        try:
            self._batch_upsert(index_name, docs)
        except Exception as e:
            logger.error(f"VBConnection._insert_doc_meta error: {str(e)}")
            res.append(str(e))
        return res

    def update(self, condition: dict, new_value: dict, index_name: str, knowledgebase_id: str) -> bool:
        """Update rows matching condition with new values.

        Args:
            condition: Filter condition dict.
            new_value: Dict of column-value pairs to set.
            index_name: Target table name.
            knowledgebase_id: Knowledge base identifier.

        Returns:
            True if update succeeded, False otherwise.
        """
        if not self._check_table_exists_cached(index_name):
            return True

        if not index_name.startswith("ragflow_doc_meta_"):
            condition["kb_id"] = knowledgebase_id
        filters = get_filters(condition)
        set_values: list[str] = []
        for k, v in new_value.items():
            if k == "remove":
                if isinstance(v, str):
                    set_values.append(f"\"{v}\" = NULL")
                else:
                    if not isinstance(v, dict):
                        raise ValueError(f"Expected str or dict for 'remove', got {type(v)}.")
                    for kk, vv in v.items():
                        if kk not in array_columns:
                            raise ValueError(f"Column '{kk}' is not an array column.")
                        # ADAPT: PostgreSQL array_remove replaces MySQL array_remove
                        set_values.append(
                            f"\"{kk}\" = array_remove(\"{kk}\", {get_value_str(vv)})"
                        )
            elif k == "add":
                if not isinstance(v, dict):
                    raise ValueError(f"Expected dict for 'add', got {type(v)}.")
                for kk, vv in v.items():
                    if kk not in array_columns:
                        raise ValueError(f"Column '{kk}' is not an array column.")
                    # ADAPT: PostgreSQL array_append replaces MySQL array_append
                    set_values.append(
                        f"\"{kk}\" = array_append(\"{kk}\", {get_value_str(vv)})"
                    )
            elif k == "metadata":
                if not isinstance(v, dict):
                    raise ValueError(f"Expected dict for 'metadata', got {type(v)}")
                set_values.append(f"\"{k}\" = {get_value_str(v)}")
                if v and "doc_id" in condition:
                    group_id = v.get("_group_id")
                    title = v.get("_title")
                    if group_id:
                        set_values.append(f"\"group_id\" = {get_value_str(group_id)}")
                    if title:
                        set_values.append(f"\"docnm_kwd\" = {get_value_str(title)}")
            else:
                set_values.append(f"\"{k}\" = {get_value_str(v)}")

        if not set_values:
            return True

        update_sql = (
            f"UPDATE \"{index_name}\""
            f" SET {', '.join(set_values)}"
            f" WHERE {' AND '.join(filters)}"
        )
        logger.debug("VBConnection.update sql: %s", update_sql)

        try:
            self._execute_sql(update_sql)
            return True
        except Exception as e:
            logger.error(f"VBConnection.update error: {str(e)}")
        return False

    def _row_to_entity(self, data, fields: list[str]) -> dict:
        entity = {}
        for i, field in enumerate(fields):
            value = data[i]
            if value is None:
                continue
            entity[field] = get_column_value(field, value)
        return entity

    """
    Helper functions for search result
    """

    def get_total(self, res) -> int:
        return res.total

    def get_doc_ids(self, res) -> list[str]:
        return [row["id"] for row in res.chunks]

    def get_fields(self, res, fields: list[str]) -> dict[str, dict]:
        result = {}
        for row in res.chunks:
            data = {}
            for field in fields:
                v = row.get(field)
                if v is not None:
                    data[field] = v
            result[row["id"]] = data
        return result

    def is_chinese(self, line):
        arr = re.split(r"[ \t]+", line)
        if len(arr) <= 3:
            return True
        e = 0
        for t in arr:
            if not re.match(r"[a-zA-Z]+$", t):
                e += 1
        return e * 1.0 / len(arr) >= 0.7

    def highlight(self, txt: str, tks: str, question: str, keywords: list[str]) -> Optional[str]:
        if not txt or not keywords:
            return None

        highlighted_txt = txt

        if question and not self.is_chinese(question):
            highlighted_txt = re.sub(
                r"(^|\W)(%s)(\W|$)" % re.escape(question),
                r"\1<em>\2</em>\3", highlighted_txt,
                flags=re.IGNORECASE | re.MULTILINE,
            )
            if re.search(r"<em>[^<>]+</em>", highlighted_txt, flags=re.IGNORECASE | re.MULTILINE):
                return highlighted_txt

            for keyword in keywords:
                highlighted_txt = re.sub(
                    r"(^|\W)(%s)(\W|$)" % re.escape(keyword),
                    r"\1<em>\2</em>\3", highlighted_txt,
                    flags=re.IGNORECASE | re.MULTILINE,
                )
            if len(re.findall(r'</em><em>', highlighted_txt)) > 0 or len(
                    re.findall(r'</em>\s*<em>', highlighted_txt)) > 0:
                return highlighted_txt
            else:
                return None

        if not tks:
            tks = rag_tokenizer.tokenize(txt)
        tokens = tks.split()
        if not tokens:
            return None

        last_pos = len(txt)
        for i in range(len(tokens) - 1, -1, -1):
            token = tokens[i]
            token_pos = highlighted_txt.rfind(token, 0, last_pos)
            if token_pos != -1:
                if token in keywords:
                    highlighted_txt = (
                            highlighted_txt[:token_pos] +
                            f'<em>{token}</em>' +
                            highlighted_txt[token_pos + len(token):]
                    )
                last_pos = token_pos
        return re.sub(r'</em><em>', '', highlighted_txt)

    def get_highlight(self, res, keywords: list[str], fieldnm: str):
        ans = {}
        if len(res.chunks) == 0 or len(keywords) == 0:
            return ans

        for d in res.chunks:
            txt = d.get(fieldnm)
            if not txt:
                continue

            tks = d.get("content_ltks") if fieldnm == "content_with_weight" else ""
            highlighted_txt = self.highlight(txt, tks, " ".join(keywords), keywords)
            if highlighted_txt:
                ans[d["id"]] = highlighted_txt
        return ans

    def get_aggregation(self, res, fieldnm: str):
        if len(res.chunks) == 0:
            return []

        counts = {}
        result = []
        for d in res.chunks:
            if "value" in d and "count" in d:
                result.append((d["value"], d["count"]))
            elif fieldnm in d:
                values = d[fieldnm]
                if isinstance(values, str):
                    try:
                        values = json.loads(values)
                    except json.JSONDecodeError:
                        values = [values]
                elif not isinstance(values, list):
                    values = [values]

                if isinstance(values, list):
                    for vv in values:
                        if isinstance(vv, str):
                            counts[vv] = counts.get(vv, 0) + 1
            else:
                continue

        if counts:
            result = list(counts.items())

        result.sort(key=lambda x: x[1] * -1)
        return result
