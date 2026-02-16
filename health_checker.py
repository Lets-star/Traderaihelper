"""
Health check system for monitoring API connections, credentials, and system status.

This module provides comprehensive health checking for:
- API connectivity
- Credential validation
- WebSocket connections
- System resources
- Database connectivity
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests

from models import HealthCheck, HealthStatus, Credentials

logger = logging.getLogger(__name__)


class HealthChecker:
    """
    Comprehensive health checking system.

    Provides health checks for various system components including
    API connectivity, credentials, and system resources.
    """

    def __init__(self, check_interval: int = 60):
        """
        Initialize health checker.

        Args:
            check_interval: Interval between automatic health checks (seconds)
        """
        self._lock = threading.RLock()
        self.check_interval = check_interval
        self._last_checks: Dict[str, HealthCheck] = {}
        self._auto_check_enabled = False
        self._auto_check_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def check_api_connection(
        self,
        base_url: str,
        timeout: int = 10,
        endpoint: str = "/v5/market/tickers?category=linear"
    ) -> HealthCheck:
        """
        Check API connectivity.

        Args:
            base_url: API base URL
            timeout: Request timeout in seconds
            endpoint: Health check endpoint

        Returns:
            HealthCheck object with results
        """
        component = f"api_connection_{base_url}"
        start_time = time.time()

        try:
            url = f"{base_url}{endpoint}"
            response = requests.get(url, timeout=timeout)
            response_time_ms = (time.time() - start_time) * 1000

            if response.status_code == 200:
                status = HealthStatus.HEALTHY
                message = "API connection successful"
                details = {
                    "status_code": response.status_code,
                    "response_time_ms": response_time_ms,
                }
            elif response.status_code == 429:
                status = HealthStatus.DEGRADED
                message = "API rate limit hit"
                details = {
                    "status_code": response.status_code,
                    "response_time_ms": response_time_ms,
                }
            else:
                status = HealthStatus.UNHEALTHY
                message = f"API returned status {response.status_code}"
                details = {
                    "status_code": response.status_code,
                    "response_time_ms": response_time_ms,
                    "response_text": response.text[:200],
                }

        except requests.exceptions.Timeout:
            status = HealthStatus.UNHEALTHY
            message = f"API connection timeout after {timeout}s"
            details = {"timeout": timeout}
            response_time_ms = timeout * 1000

        except requests.exceptions.ConnectionError as e:
            status = HealthStatus.UNHEALTHY
            message = f"API connection error: {str(e)}"
            details = {"error": str(e)}
            response_time_ms = (time.time() - start_time) * 1000

        except Exception as e:
            status = HealthStatus.UNHEALTHY
            message = f"Unexpected API error: {str(e)}"
            details = {"error": str(e)}
            response_time_ms = (time.time() - start_time) * 1000

        health_check = HealthCheck(
            component=component,
            status=status,
            message=message,
            details=details,
            response_time_ms=response_time_ms,
        )

        with self._lock:
            self._last_checks[component] = health_check

        logger.debug(f"API health check: {component} - {status.value} - {message}")
        return health_check

    def check_credentials(
        self,
        credentials: Credentials,
        timeout: int = 10
    ) -> HealthCheck:
        """
        Check API credentials by attempting to fetch wallet balance.

        Args:
            credentials: Credentials object
            timeout: Request timeout in seconds

        Returns:
            HealthCheck object with results
        """
        component = f"credentials_{credentials.exchange}"

        # First, validate credentials structure
        try:
            if not credentials.is_valid():
                status = HealthStatus.UNHEALTHY
                message = "Invalid credentials structure"
                details = {
                    "api_key_length": len(credentials.api_key),
                    "api_secret_length": len(credentials.api_secret),
                }
                health_check = HealthCheck(
                    component=component,
                    status=status,
                    message=message,
                    details=details,
                )
                with self._lock:
                    self._last_checks[component] = health_check
                return health_check
        except Exception as e:
            status = HealthStatus.UNHEALTHY
            message = f"Credential validation error: {str(e)}"
            details = {"error": str(e)}
            health_check = HealthCheck(
                component=component,
                status=status,
                message=message,
                details=details,
            )
            with self._lock:
                self._last_checks[component] = health_check
            return health_check

        # Test credentials via API call
        start_time = time.time()

        if credentials.exchange == "bybit":
            base_url = (
                "https://api-testnet.bybit.com" if credentials.testnet
                else "https://api.bybit.com"
            )

            try:
                # Import here to avoid circular imports
                from bybit_client import ByBitClient

                with ByBitClient(
                    credentials.api_key,
                    credentials.api_secret,
                    testnet=credentials.testnet,
                    log_trades=False
                ) as client:
                    result = client.get_wallet_balance(account_type="UNIFIED")

                response_time_ms = (time.time() - start_time) * 1000

                if result.get("retCode") == 0:
                    status = HealthStatus.HEALTHY
                    message = "Credentials valid and working"
                    details = {
                        "account_type": "UNIFIED",
                        "response_time_ms": response_time_ms,
                    }
                else:
                    status = HealthStatus.UNHEALTHY
                    message = f"API rejected credentials: {result.get('retMsg', 'Unknown error')}"
                    details = {
                        "ret_code": result.get("retCode"),
                        "ret_msg": result.get("retMsg"),
                        "response_time_ms": response_time_ms,
                    }

            except Exception as e:
                status = HealthStatus.UNHEALTHY
                message = f"Credential test failed: {str(e)}"
                details = {"error": str(e)}
                response_time_ms = (time.time() - start_time) * 1000

        elif credentials.exchange == "binance":
            # Similar logic for Binance
            status = HealthStatus.UNKNOWN
            message = "Binance credential check not implemented"
            details = {}
            response_time_ms = (time.time() - start_time) * 1000

        else:
            status = HealthStatus.UNKNOWN
            message = f"Credential check not implemented for {credentials.exchange}"
            details = {}
            response_time_ms = 0.0

        health_check = HealthCheck(
            component=component,
            status=status,
            message=message,
            details=details,
            response_time_ms=response_time_ms,
        )

        with self._lock:
            self._last_checks[component] = health_check

        logger.info(f"Credentials health check: {component} - {status.value} - {message}")
        return health_check

    def check_websocket(
        self,
        symbol: str,
        interval: str = "1m"
    ) -> HealthCheck:
        """
        Check WebSocket connectivity.

        Args:
            symbol: Trading symbol
            interval: Candle interval

        Returns:
            HealthCheck object with results
        """
        component = f"websocket_{symbol}_{interval}"

        try:
            # Import here to avoid circular imports
            from websocket_client import BinanceWebSocketClient

            # Create temporary client to test connection
            connected_event = threading.Event()

            def on_open(ws):
                connected_event.set()

            client = BinanceWebSocketClient(
                symbol=symbol,
                interval=interval,
                on_closed_bar=None,
                on_forming_bar=None
            )

            # Override on_open temporarily
            client._on_open = on_open

            start_time = time.time()
            client.start()

            # Wait up to 5 seconds for connection
            connected = connected_event.wait(timeout=5.0)
            response_time_ms = (time.time() - start_time) * 1000

            client.stop()

            if connected:
                status = HealthStatus.HEALTHY
                message = "WebSocket connection successful"
                details = {
                    "symbol": symbol,
                    "interval": interval,
                    "response_time_ms": response_time_ms,
                }
            else:
                status = HealthStatus.DEGRADED
                message = "WebSocket connection timeout"
                details = {
                    "symbol": symbol,
                    "interval": interval,
                    "timeout_seconds": 5.0,
                }

        except Exception as e:
            status = HealthStatus.UNHEALTHY
            message = f"WebSocket check failed: {str(e)}"
            details = {"error": str(e)}
            response_time_ms = (time.time() - start_time) * 1000 if 'start_time' in locals() else 0.0

        health_check = HealthCheck(
            component=component,
            status=status,
            message=message,
            details=details,
            response_time_ms=response_time_ms,
        )

        with self._lock:
            self._last_checks[component] = health_check

        logger.debug(f"WebSocket health check: {component} - {status.value} - {message}")
        return health_check

    def check_system_resources(self) -> HealthCheck:
        """
        Check system resources (memory, CPU, etc.).

        Returns:
            HealthCheck object with results
        """
        component = "system_resources"

        try:
            import psutil

            # Memory info
            memory = psutil.virtual_memory()
            memory_percent = memory.percent

            # CPU info
            cpu_percent = psutil.cpu_percent(interval=1.0)

            # Disk info
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent

            # Determine status
            status = HealthStatus.HEALTHY
            issues = []

            if memory_percent > 90:
                status = HealthStatus.UNHEALTHY
                issues.append(f"High memory usage: {memory_percent:.1f}%")
            elif memory_percent > 75:
                status = HealthStatus.DEGRADED
                issues.append(f"Elevated memory usage: {memory_percent:.1f}%")

            if cpu_percent > 90:
                status = HealthStatus.UNHEALTHY
                issues.append(f"High CPU usage: {cpu_percent:.1f}%")
            elif cpu_percent > 75:
                status = max(status, HealthStatus.DEGRADED)  # Don't downgrade to HEALTHY
                issues.append(f"Elevated CPU usage: {cpu_percent:.1f}%")

            if disk_percent > 90:
                status = HealthStatus.UNHEALTHY
                issues.append(f"High disk usage: {disk_percent:.1f}%")
            elif disk_percent > 80:
                status = max(status, HealthStatus.DEGRADED)
                issues.append(f"Elevated disk usage: {disk_percent:.1f}%")

            if status == HealthStatus.HEALTHY:
                message = "System resources healthy"
            else:
                message = "; ".join(issues)

            details = {
                "memory_percent": memory_percent,
                "memory_available_gb": memory.available / (1024**3),
                "cpu_percent": cpu_percent,
                "disk_percent": disk_percent,
                "disk_free_gb": disk.free / (1024**3),
            }

        except ImportError:
            status = HealthStatus.UNKNOWN
            message = "psutil not installed, skipping resource check"
            details = {}

        except Exception as e:
            status = HealthStatus.UNHEALTHY
            message = f"Resource check failed: {str(e)}"
            details = {"error": str(e)}

        health_check = HealthCheck(
            component=component,
            status=status,
            message=message,
            details=details,
        )

        with self._lock:
            self._last_checks[component] = health_check

        logger.debug(f"System resources health check: {status.value} - {message}")
        return health_check

    def run_all_checks(self) -> Dict[str, HealthCheck]:
        """
        Run all configured health checks.

        Returns:
            Dictionary mapping component names to HealthCheck objects
        """
        results = {}

        # Check system resources
        try:
            results["system_resources"] = self.check_system_resources()
        except Exception as e:
            logger.error(f"System resources check failed: {e}")

        return results

    def get_last_check(self, component: str) -> Optional[HealthCheck]:
        """
        Get the last health check result for a component.

        Args:
            component: Component name

        Returns:
            HealthCheck object or None if no check has been performed
        """
        with self._lock:
            return self._last_checks.get(component)

    def get_all_last_checks(self) -> Dict[str, HealthCheck]:
        """
        Get all last health check results.

        Returns:
            Dictionary mapping component names to HealthCheck objects
        """
        with self._lock:
            return self._last_checks.copy()

    def clear_checks(self) -> None:
        """Clear all stored health check results."""
        with self._lock:
            self._last_checks.clear()

    def start_auto_check(self, credentials: Optional[Credentials] = None) -> None:
        """
        Start automatic health checking in background thread.

        Args:
            credentials: Optional credentials to check
        """
        with self._lock:
            if self._auto_check_enabled:
                logger.warning("Auto health check already running")
                return

            self._auto_check_enabled = True
            self._stop_event.clear()

            self._auto_check_thread = threading.Thread(
                target=self._auto_check_loop,
                args=(credentials,),
                daemon=True,
            )
            self._auto_check_thread.start()
            logger.info("Started automatic health checking")

    def stop_auto_check(self) -> None:
        """Stop automatic health checking."""
        with self._lock:
            if not self._auto_check_enabled:
                return

            self._stop_event.set()
            self._auto_check_enabled = False

        if self._auto_check_thread:
            self._auto_check_thread.join(timeout=5.0)
            self._auto_check_thread = None

        logger.info("Stopped automatic health checking")

    def _auto_check_loop(self, credentials: Optional[Credentials]) -> None:
        """Background thread loop for automatic health checking."""
        while not self._stop_event.is_set():
            try:
                # Run all checks
                if credentials:
                    try:
                        self.check_credentials(credentials)
                    except Exception as e:
                        logger.error(f"Auto credential check failed: {e}")

                try:
                    self.check_system_resources()
                except Exception as e:
                    logger.error(f"Auto system check failed: {e}")

                # Wait for next check interval or stop event
                self._stop_event.wait(self.check_interval)

            except Exception as e:
                logger.error(f"Error in auto health check loop: {e}")
                # Wait a bit before retrying
                self._stop_event.wait(min(self.check_interval, 30))

    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all health check results.

        Returns:
            Dictionary with health summary
        """
        with self._lock:
            checks = self._last_checks.copy()

        if not checks:
            return {
                "status": "unknown",
                "message": "No health checks performed",
                "components": {},
            }

        healthy_count = sum(1 for c in checks.values() if c.status == HealthStatus.HEALTHY)
        degraded_count = sum(1 for c in checks.values() if c.status == HealthStatus.DEGRADED)
        unhealthy_count = sum(1 for c in checks.values() if c.status == HealthStatus.UNHEALTHY)
        unknown_count = sum(1 for c in checks.values() if c.status == HealthStatus.UNKNOWN)

        # Determine overall status
        if unhealthy_count > 0:
            overall_status = "unhealthy"
            message = f"{unhealthy_count} component(s) unhealthy"
        elif degraded_count > 0:
            overall_status = "degraded"
            message = f"{degraded_count} component(s) degraded"
        elif unknown_count > 0 and healthy_count == 0:
            overall_status = "unknown"
            message = "No components checked"
        else:
            overall_status = "healthy"
            message = f"All {healthy_count} component(s) healthy"

        return {
            "status": overall_status,
            "message": message,
            "components": {
                name: check.to_dict()
                for name, check in checks.items()
            },
            "counts": {
                "healthy": healthy_count,
                "degraded": degraded_count,
                "unhealthy": unhealthy_count,
                "unknown": unknown_count,
                "total": len(checks),
            },
        }
