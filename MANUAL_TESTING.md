# RAGFlow Vastbase 适配 — 人工测试教程

> **注意**: 此为配置注册阶段（Task 3/VAS-21）测试教程。完整端到端测试教程将在父 Issue VAS-19 集成验收阶段完成。

## 环境要求

- Python 版本：>=3.13, <3.15
- Vastbase G100 版本：V3 (3.0.8+)
- pyvastbase 版本：0.2.6+
- 操作系统：macOS / Linux

## Vastbase 连接信息

确保 Vastbase 实例已启动且 pgvector 扩展已启用。

```bash
export VASTBASE_HOST=172.16.105.107
export VASTBASE_PORT=15432
export VASTBASE_USER=aidev
export VASTBASE_PASSWORD=Vbase_123456
export VASTBASE_DATABASE=vastbase
```

## 安装步骤

### 1. 克隆仓库

```bash
git clone https://github.com/Loger0/ragflow.git
cd ragflow
git checkout feature/ragflow-vastbase-backend
```

### 2. 安装依赖

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install pyvastbase pytest pyyaml
```

### 3. 验证配置注册

```bash
# 检查 vastbase 配置节
grep -A6 "vastbase:" conf/service_conf.yaml

# 检查环境变量
grep "VASTBASE" docker/.env
```

预期输出：vastbase 配置节包含 host/port/db_name/user/password/max_connections。

## 验证步骤

### 配置注册验证

```bash
pytest tests/test_vas21_config_registration.py -v -c /dev/null
```

预期：43 项测试全部 PASS。

### pyvastbase 连接测试

```bash
python -c "
from pyvastbase import connect, health_check
connect(host='172.16.105.107', port=15432, dbname='vastbase', user='aidev', password='Vbase_123456')
print(health_check())
"
```

预期输出：`{'status': 'healthy', 'connected': True, ...}`

### 配置注册检查清单

| 检查项 | 文件 | 状态 |
|--------|------|------|
| DOC_ENGINE_VASTBASE 常量 | `common/settings.py:90` | ✅ |
| VASTBASE 字典初始化 | `common/settings.py:126` | ✅ |
| docStoreConn elif 分支 | `common/settings.py:329-331` | ✅ |
| msgStoreConn elif 分支 | `common/settings.py:349-350` | ✅ |
| vastbase 配置节 | `conf/service_conf.yaml:46-52` | ✅ |
| VASTBASE_* 环境变量 | `docker/.env:111-119` | ✅ |
| vastbase 模板配置节 | `docker/service_conf.yaml.template:58-64` | ✅ |

## 常见问题

### pyvastbase 连接失败
检查 Vastbase 实例是否启动、端口是否可达、凭据是否正确。
```bash
nc -zv 172.16.105.107 15432
```

### RAGFlow 全量测试依赖问题
RAGFlow 全量测试需要 Flask、多种 LLM SDK、Elasticsearch 等大量依赖。配置注册阶段测试使用最小依赖集（pyvastbase + pytest + pyyaml）。
