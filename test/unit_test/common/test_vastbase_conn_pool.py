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
Unit tests for Vastbase connection pool (vastbase_conn_pool.py).
"""
import sys
import pytest
from unittest.mock import Mock, patch


def _setup_mocks():
    """Set up mock modules in sys.modules before importing vastbase_conn_pool."""
    mock_pyvastbase = Mock()
    mock_pyvastbase.connect = Mock(return_value=Mock())
    mock_pyvastbase.get_connection = Mock(return_value=Mock())
    mock_pyvastbase.health_check = Mock(return_value={"status": "healthy"})

    mock_settings = Mock()
    mock_settings.VASTBASE = {
        "host": "test-host",
        "port": 15432,
        "user": "test-user",
        "password": "test-pass",
        "db_name": "test-db",
        "max_connections": 100,
    }

    # Clear cached modules for fresh import
    for key in list(sys.modules.keys()):
        if key.startswith('common.doc_store.vastbase_conn_pool'):
            del sys.modules[key]
        elif key == 'common.decorator':
            del sys.modules[key]

    sys.modules['pyvastbase'] = mock_pyvastbase
    sys.modules['common.settings'] = mock_settings

    return mock_pyvastbase, mock_settings


def _teardown_mocks():
    """Remove mock modules from sys.modules."""
    for key in list(sys.modules.keys()):
        if key.startswith('common.doc_store.vastbase_conn_pool'):
            del sys.modules[key]
        elif key == 'common.decorator':
            del sys.modules[key]


class TestVastbaseConnectionPool:
    """Test cases for VastbaseConnectionPool class."""

    def test_init_with_settings_vastbase_attr(self):
        """Test initialization with settings.VASTBASE defined."""
        mock_pyvastbase, _ = _setup_mocks()
        try:
            import common.doc_store.vastbase_conn_pool as vcp
            pool = vcp.VastbaseConnectionPool()

            assert pool.db_name == "test-db"
            assert pool.uri == "test-host:15432"
            mock_pyvastbase.connect.assert_called()
        finally:
            _teardown_mocks()

    def test_init_with_get_base_config_fallback(self):
        """Test initialization when VASTBASE attr not on settings (fallback to get_base_config)."""
        mock_pyvastbase, mock_settings = _setup_mocks()
        del mock_settings.VASTBASE
        mock_settings.get_base_config.return_value = {
            "host": "fallback-host",
            "port": 16000,
            "user": "fb-user",
            "password": "fb-pass",
            "db_name": "fb-db",
            "max_connections": 50,
        }
        try:
            import common.doc_store.vastbase_conn_pool as vcp
            pool = vcp.VastbaseConnectionPool()

            assert pool.db_name == "fb-db"
            assert pool.uri == "fallback-host:16000"
            mock_settings.get_base_config.assert_called_once_with("vastbase", {})
        finally:
            _teardown_mocks()

    def test_retry_on_first_attempt_fail(self):
        """Test ATTEMPT_TIME=2 retry: first attempt fails, second succeeds."""
        mock_pyvastbase, _ = _setup_mocks()
        mock_client = Mock()
        mock_pyvastbase.connect.side_effect = [Exception("Connection refused"), mock_client]

        # Need to mock time.sleep BEFORE module import (VB_CONN created at import time)
        import time as real_time
        try:
            with patch.object(real_time, 'sleep') as mock_sleep:
                import common.doc_store.vastbase_conn_pool as vcp

            assert mock_pyvastbase.connect.call_count == 2
            mock_sleep.assert_called_once_with(5)
        finally:
            _teardown_mocks()

    def test_raises_after_all_retries_fail(self):
        """Test exception raised after ATTEMPT_TIME=2 failures at module import."""
        mock_pyvastbase, _ = _setup_mocks()
        mock_pyvastbase.connect.side_effect = Exception("Connection refused")

        import time as real_time
        try:
            with patch.object(real_time, 'sleep'), \
                 pytest.raises(Exception, match="connection failed after 2 attempts"):
                import common.doc_store.vastbase_conn_pool as vcp
        finally:
            _teardown_mocks()

    def test_health_check_raises_on_failure(self):
        """Test that _check_vastbase_health raises when health_check fails at import."""
        mock_pyvastbase, _ = _setup_mocks()
        # connect succeeds but health_check fails
        mock_pyvastbase.health_check.side_effect = Exception("Health check failed")

        try:
            with pytest.raises(Exception, match="Failed to check Vastbase health"):
                import common.doc_store.vastbase_conn_pool as vcp
        finally:
            _teardown_mocks()

    def test_get_client_returns_connection(self):
        """Test get_client returns the pyvastbase connection."""
        mock_pyvastbase, _ = _setup_mocks()
        mock_client = Mock()
        mock_pyvastbase.connect.return_value = mock_client
        try:
            import common.doc_store.vastbase_conn_pool as vcp
            pool = vcp.VastbaseConnectionPool()
            assert pool.get_client() is mock_client
        finally:
            _teardown_mocks()

    def test_get_db_name(self):
        """Test get_db_name returns correct value."""
        _setup_mocks()
        try:
            import common.doc_store.vastbase_conn_pool as vcp
            pool = vcp.VastbaseConnectionPool()
            assert pool.get_db_name() == "test-db"
        finally:
            _teardown_mocks()

    def test_get_uri(self):
        """Test get_uri returns correct value."""
        _setup_mocks()
        try:
            import common.doc_store.vastbase_conn_pool as vcp
            pool = vcp.VastbaseConnectionPool()
            assert pool.get_uri() == "test-host:15432"
        finally:
            _teardown_mocks()

    def test_refresh_client_healthy(self):
        """Test refresh_client returns client when SELECT 1 succeeds."""
        mock_pyvastbase, _ = _setup_mocks()
        mock_client = Mock()
        mock_cursor = Mock()
        mock_client.cursor.return_value = mock_cursor
        mock_pyvastbase.connect.return_value = mock_client
        try:
            import common.doc_store.vastbase_conn_pool as vcp
            pool = vcp.VastbaseConnectionPool()
            result = pool.refresh_client()
            assert result is mock_client
            mock_client.cursor.assert_called()
            mock_cursor.execute.assert_called_with("SELECT 1")
            mock_cursor.close.assert_called_once()
        finally:
            _teardown_mocks()

    def test_refresh_client_reconnect_on_failure(self):
        """Test refresh_client reconnects when SELECT 1 fails."""
        mock_pyvastbase, _ = _setup_mocks()
        mock_client = Mock()
        mock_cursor = Mock()
        mock_cursor.execute.side_effect = Exception("Connection lost")
        mock_client.cursor.return_value = mock_cursor
        mock_pyvastbase.connect.return_value = mock_client
        new_client = Mock()
        mock_pyvastbase.get_connection.return_value = new_client
        try:
            import common.doc_store.vastbase_conn_pool as vcp
            pool = vcp.VastbaseConnectionPool()
            result = pool.refresh_client()
            assert result is new_client
            mock_client.close.assert_called_once()
        finally:
            _teardown_mocks()

    def test_del_cleans_up_client(self):
        """Test __del__ closes connection cleanly."""
        mock_pyvastbase, _ = _setup_mocks()
        mock_client = Mock()
        mock_pyvastbase.connect.return_value = mock_client
        try:
            import common.doc_store.vastbase_conn_pool as vcp
            pool = vcp.VastbaseConnectionPool()
            pool.__del__()
            mock_client.close.assert_called_once()
        finally:
            _teardown_mocks()

    def test_health_check_uses_correct_alias(self):
        """Test that health_check is called with vastbase_doc_store alias."""
        mock_pyvastbase, _ = _setup_mocks()
        try:
            import common.doc_store.vastbase_conn_pool as vcp
            vcp.VastbaseConnectionPool()
            mock_pyvastbase.health_check.assert_called_with(using="vastbase_doc_store")
        finally:
            _teardown_mocks()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
