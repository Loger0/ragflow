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
import logging
import os
import time

from pyvastbase import connect
from pyvastbase import get_connection
from pyvastbase import health_check

from common import settings
from common.decorator import singleton

ATTEMPT_TIME = 2

logger = logging.getLogger('ragflow.vastbase_conn_pool')


@singleton
class VastbaseConnectionPool:
    """Connection pool for Vastbase document store.

    Uses pyvastbase native connection management with ATTEMPT_TIME=2 retry.
    Provides get_client(), get_db_name(), and get_uri() for downstream consumers.
    """

    def __init__(self):
        self.client = None

        if hasattr(settings, "VASTBASE"):
            self.VB_CONFIG = settings.VASTBASE
        else:
            self.VB_CONFIG = settings.get_base_config("vastbase", {})

        host = self.VB_CONFIG.get("host", "localhost")
        port = self.VB_CONFIG.get("port", 15432)
        self.user = self.VB_CONFIG.get("user", "aidev")
        self.password = self.VB_CONFIG.get("password", "Vbase_123456")
        self.db_name = self.VB_CONFIG.get("db_name", "vastbase")
        max_connections = self.VB_CONFIG.get("max_connections", 300)

        self.uri = f"{host}:{port}"

        logger.info(f"Use Vastbase '{self.uri}' as the doc engine.")

        max_overflow = int(os.environ.get("VB_MAX_OVERFLOW", str(max(max_connections // 2, 10))))
        pool_timeout = int(os.environ.get("VB_POOL_TIMEOUT", "30"))

        for attempt in range(ATTEMPT_TIME):
            try:
                # ADAPT: pyvastbase.connect() replaces ObVecClient for Vastbase/PostgreSQL wire protocol
                self.client = connect(
                    host=host,
                    port=int(port),
                    database=self.db_name,
                    user=self.user,
                    password=self.password,
                    pool_size=max_connections,
                    max_overflow=max_overflow,
                    timeout=pool_timeout,
                    alias="vastbase_doc_store",
                )
                break
            except Exception as e:
                logger.warning(f"{str(e)}. Waiting Vastbase {self.uri} to be healthy.")
                time.sleep(5)

        if self.client is None:
            msg = f"Vastbase {self.uri} connection failed after {ATTEMPT_TIME} attempts."
            logger.error(msg)
            raise Exception(msg)

        self._check_vastbase_health()
        logger.info(f"Vastbase {self.uri} is healthy.")

    def _check_vastbase_health(self):
        """Verify Vastbase connection is healthy."""
        try:
            # ADAPT: health_check() replaces MySQL SHOW VARIABLES for Vastbase
            result = health_check(using="vastbase_doc_store")
            logger.info(f"Vastbase {self.uri} health check: {result}")
        except Exception as e:
            raise Exception(f"Failed to check Vastbase health at {self.uri}, error: {str(e)}")

    def get_client(self):
        """Return the pyvastbase connection."""
        return self.client

    def get_db_name(self) -> str:
        return self.db_name

    def get_uri(self) -> str:
        return self.uri

    def refresh_client(self):
        """Refresh connection if unhealthy."""
        try:
            self.client.execute("SELECT 1")
            return self.client
        except Exception as e:
            logger.warning(f"Vastbase connection unhealthy: {str(e)}, reconnecting...")
            # ADAPT: pyvastbase reconnect — close and re-connect
            try:
                self.client.close()
            except Exception:
                pass
            self.client = get_connection("vastbase_doc_store")
            return self.client

    def __del__(self):
        if hasattr(self, "client") and self.client:
            try:
                self.client.close()
            except Exception:
                pass


VB_CONN = VastbaseConnectionPool()
