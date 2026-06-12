# RAGFlow Vastbase 适配 — 修改总结

> **阶段**: 配置注册 (Task 3 / VAS-21)。完整修改清单在父 Issue VAS-19 集成验收阶段汇总。

## 概述

将 RAGFlow 的文档/消息存储引擎从仅支持 Elasticsearch/Infinity/OpenSearch/OceanBase 扩展为同时支持 Vastbase G100 (PostgreSQL 协议 + pgvector)。

配置注册阶段（VAS-21）完成：settings.py 分发入口、YAML 配置模板、Docker 环境变量、pyvastbase 连接基础设施。

## 修改文件

| 文件 | 修改类型 | 说明 |
|------|----------|------|
| `common/settings.py` | 修改 | 添加 `DOC_ENGINE_VASTBASE` 常量、`VASTBASE` 字典初始化、vastbase elif 分支（docStoreConn + msgStoreConn） |
| `conf/service_conf.yaml` | 修改 | 新增 `vastbase:` 配置节（host/port/db_name/user/password/max_connections） |
| `docker/.env` | 修改 | 新增 `VASTBASE_HOST/PORT/USER/PASSWORD/DOC_DBNAME` 环境变量，DOC_ENGINE 注释添加 vastbase 选项 |
| `docker/service_conf.yaml.template` | 修改 | 新增 `vastbase:` 配置节（使用环境变量占位符） |

## 新增文件

| 文件 | 说明 | 行数 |
|------|------|------|
| `common/doc_store/vastbase_conn_base.py` | Vastbase 独立基类，继承 `DocStoreConnection(ABC)`，使用 PostgreSQL/Vastbase SQL 方言 | ~1000 |
| `common/doc_store/vastbase_conn_pool.py` | pyvastbase 连接池单例 (@singleton, ATTEMPT_TIME=2) | ~100 |
| `rag/utils/vastbase_conn.py` | Chunk 存储和搜索实现（继承 VastbaseConnectionBase） | ~1100 |
| `memory/utils/vastbase_conn.py` | 消息存储实现（继承 VastbaseConnectionBase） | ~500 |

## 架构决策

### 三层架构

```
DocStoreConnection (ABC)
  └── VastbaseConnectionBase (common/doc_store/vastbase_conn_base.py)
        ├── VBConnection (rag/utils/vastbase_conn.py)       ← Chunk 存储
        └── VBConnection (memory/utils/vastbase_conn.py)    ← 消息存储
```

### 关键设计选择

| # | 决策 | 选择 |
|---|------|------|
| 1 | 连接管理 | pyvastbase 原生 `connect()` + `get_connection()` |
| 2 | 注册方式 | `settings.py` 中 `if/elif` 硬编码链分发，与 OB 模式一致 |
| 3 | SQL 策略 | pyvastbase 优先，复杂场景 Vastbase SQL |
| 4 | 全文搜索 | Vastbase BM25：`bm25("col", 'query')` + `"col" @~@ 'query'` |
| 5 | 向量搜索 | pgvector `<=>` 操作符 + 标准 `LIMIT/OFFSET` |
| 6 | 错误处理 | 照搬 OB 模式（读 re-raise / 写吞异常，连接重试） |

## 与上游框架的差异

| 差异项 | OceanBase (ob_conn) | Vastbase (vastbase_conn) |
|--------|---------------------|--------------------------|
| 数据库协议 | MySQL 兼容 | PostgreSQL 协议 |
| 向量操作符 | `cosine_distance()` | pgvector `<=>` |
| 全文搜索 | `MATCH() AGAINST()` | BM25 `@~@` |
| LIMIT 语法 | `LIMIT offset, limit` | `LIMIT limit OFFSET offset` |
| 连接客户端 | ObVecClient (pyobvector) | pyvastbase |
| DDL | ObVecClient API | PostgreSQL 原生 SQL |
| JSON 操作 | MySQL JSON 函数 | PostgreSQL JSONB `@>` / `->>` |

## 测试覆盖

| 层级 | 用例数 | 来源 | 结果 |
|------|--------|------|------|
| 单元测试（类结构） | 9 | test-adapter | ✅ 9/9 |
| SQL 方言验证 | 6 | test-adapter | ✅ 6/6 |
| 配置注册验证 | 6 | test-adapter | ✅ 6/6 |
| 配置文件验证 | 5 | test-adapter | ✅ 5/5 |
| pyvastbase 集成测试 | 5 | test-adapter | ✅ 5/5 |
| 文件结构与回归 | 5 | test-adapter | ✅ 5/5 |
| 语法与导入验证 | 7 | test-adapter | ✅ 7/7 |
| **合计** | **43** | — | **✅ 43/43** |

### 后续测试（父 Issue VAS-19 集成验收阶段）

| 层级 | 状态 |
|------|------|
| 框架官方测试（framework-tests/） | ⏳ 待 test-scout 交付 |
| 框架级集成验收 | ⏳ 待所有子 Issue done |
| 应用级 Demo | ⏳ 待 VAS-25 完成 |
