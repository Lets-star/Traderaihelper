"""
Protocol types for duck typing support.

Protocols define structural interfaces that classes can implement
without explicit inheritance, enabling better type checking for:
- Streamlit UI components
- Callback functions
- Session state access
"""

from __future__ import annotations

from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Tuple,
    Union,
    runtime_checkable,
)

from trader_types.typed_dict import SignalPayload, ExecutionResult, KlineData


@runtime_checkable
class StreamlitComponent(Protocol):
    """Protocol for Streamlit UI components (st module or containers)."""

    def number_input(
        self,
        label: str,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        value: Union[float, int, None] = None,
        step: Optional[Union[float, int]] = None,
        format: Optional[str] = None,
        key: Optional[str] = None,
        help: Optional[str] = None,
        disabled: bool = False,
        label_visibility: str = "visible",
    ) -> Union[float, int]:
        """Render number input widget."""
        ...

    def selectbox(
        self,
        label: str,
        options: List[Any],
        index: int = 0,
        format_func: Optional[Callable[[Any], str]] = None,
        key: Optional[str] = None,
        help: Optional[str] = None,
        disabled: bool = False,
        label_visibility: str = "visible",
    ) -> Any:
        """Render selectbox widget."""
        ...

    def button(
        self,
        label: str,
        key: Optional[str] = None,
        help: Optional[str] = None,
        disabled: bool = False,
        type: str = "secondary",
        use_container_width: bool = False,
    ) -> bool:
        """Render button widget."""
        ...

    def checkbox(
        self,
        label: str,
        value: bool = False,
        key: Optional[str] = None,
        help: Optional[str] = None,
        disabled: bool = False,
        label_visibility: str = "visible",
    ) -> bool:
        """Render checkbox widget."""
        ...

    def text_input(
        self,
        label: str,
        value: str = "",
        max_chars: Optional[int] = None,
        key: Optional[str] = None,
        type: str = "default",
        help: Optional[str] = None,
        disabled: bool = False,
        label_visibility: str = "visible",
    ) -> str:
        """Render text input widget."""
        ...

    def slider(
        self,
        label: str,
        min_value: Optional[Union[int, float]] = None,
        max_value: Optional[Union[int, float]] = None,
        value: Optional[Union[int, float, List[Union[int, float]]]] = None,
        step: Optional[Union[int, float]] = None,
        format: Optional[str] = None,
        key: Optional[str] = None,
        help: Optional[str] = None,
        disabled: bool = False,
        label_visibility: str = "visible",
    ) -> Union[int, float, List[Union[int, float]]]:
        """Render slider widget."""
        ...

    def radio(
        self,
        label: str,
        options: List[Any],
        index: int = 0,
        format_func: Optional[Callable[[Any], str]] = None,
        key: Optional[str] = None,
        help: Optional[str] = None,
        disabled: bool = False,
        label_visibility: str = "visible",
    ) -> Any:
        """Render radio button widget."""
        ...


@runtime_checkable
class SessionState(Protocol):
    """Protocol for Streamlit session state access."""

    def __getitem__(self, key: str) -> Any:
        """Get value by key."""
        ...

    def __setitem__(self, key: str, value: Any) -> None:
        """Set value by key."""
        ...

    def __contains__(self, key: str) -> bool:
        """Check if key exists."""
        ...

    def get(self, key: str, default: Any = None) -> Any:
        """Get value with default."""
        ...


@runtime_checkable
class UpdateCallback(Protocol):
    """Protocol for update bus callback functions."""

    def __call__(self, update: Dict[str, Any]) -> None:
        """Process an update from the bus."""
        ...


@runtime_checkable
class KlineCallback(Protocol):
    """Protocol for WebSocket kline callbacks."""

    def __call__(self, kline: KlineData) -> None:
        """Process a kline update."""
        ...


@runtime_checkable
class SignalExecutorProtocol(Protocol):
    """Protocol for signal executor implementations."""

    def execute_signal(self, signal: SignalPayload) -> None:
        """Execute a signal asynchronously."""
        ...

    def execute_signal_sync(self, signal: SignalPayload) -> ExecutionResult:
        """Execute a signal synchronously."""
        ...

    def get_position(self, symbol: str) -> Dict[str, Any]:
        """Get current position for symbol."""
        ...

    def is_position_open(self, symbol: str) -> bool:
        """Check if position is open."""
        ...

    def configure(
        self,
        enabled: bool,
        api_key: str = "",
        api_secret: str = "",
        testnet: bool = True,
        leverage: int = 5,
        pos_size_multiplier: float = 1.0,
        dry_run: bool = False,
    ) -> None:
        """Configure the executor."""
        ...


@runtime_checkable
class ChartRenderer(Protocol):
    """Protocol for chart rendering components."""

    def create_candlestick_chart(
        self,
        df: Any,  # DataFrame
        title: str,
        show_indicators: bool = True,
        height: int = 600,
    ) -> Any:  # Plotly figure
        """Create a candlestick chart."""
        ...

    def add_indicators(
        self,
        fig: Any,
        indicators: Dict[str, Any],
    ) -> Any:
        """Add indicators to chart."""
        ...


@runtime_checkable
class DataSource(Protocol):
    """Protocol for data sources (Binance, ByBit, etc.)."""

    def fetch_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 500,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch kline/candlestick data."""
        ...

    def get_server_time(self) -> int:
        """Get server time in milliseconds."""
        ...

    def is_available(self) -> bool:
        """Check if data source is available."""
        ...
