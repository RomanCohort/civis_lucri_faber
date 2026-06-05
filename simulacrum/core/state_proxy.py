"""StateProxy — lightweight read/write gate around the agent's internal state.

Each brain region registers as a *writer* for a specific set of keys.
When `strict=True`, writes to unregistered keys raise an error.
When `strict=False`, writes are allowed but logged.

Usage:
    proxy = StateProxy(internal_state, name="agent", strict=False)
    proxy.register_writer("basal_ganglia", ["bg_action", "bg_dopamine"])
    proxy.write("basal_ganglia", "bg_action", 2)          # OK
    proxy.write("basal_ganglia", "unknown_key", 42)       # logged/warned
    proxy.read("bg_action")                                # 2
"""
import logging
from typing import Any

logger = logging.getLogger(__name__)


class StateProxy:
    """Read/write gate around a shared dictionary (the internal state).

    Attributes:
        _state: reference to the backing dict
        _writers: mapping region_name -> set of allowed keys
        _name: identifier for logging
        _strict: if True, writes to unregistered keys raise KeyError
    """

    def __init__(self, state: dict[str, Any], name: str = "proxy",
                 strict: bool = False):
        self._state = state
        self._name = name
        self._strict = strict
        self._writers: dict[str, set[str]] = {}

    # ------------------------------------------------------------------
    # Writer registration
    # ------------------------------------------------------------------

    def register_writer(self, region_name: str, keys: list[str]) -> None:
        """Register a region as an allowed writer for *keys*."""
        self._writers[region_name] = set(keys)

    # ------------------------------------------------------------------
    # Read / Write
    # ------------------------------------------------------------------

    def read(self, key: str, default: Any = None) -> Any:
        """Read a value from the internal state."""
        return self._state.get(key, default)

    def write(self, region_name: str, key: str, value: Any) -> None:
        """Write a value.  If strict and key not registered, raises KeyError."""
        allowed = self._writers.get(region_name, set())
        if key not in allowed:
            msg = (f"[{self._name}] region '{region_name}' writes to "
                   f"unregistered key '{key}'")
            if self._strict:
                raise KeyError(msg)
            else:
                logger.debug(msg)
        self._state[key] = value

    # ------------------------------------------------------------------
    # Views
    # ------------------------------------------------------------------

    def view(self, keys: list[str] | None = None) -> dict[str, Any]:
        """Return a read-only snapshot of selected keys (or all if keys=None)."""
        if keys is None:
            return dict(self._state)
        return {k: self._state[k] for k in keys if k in self._state}

    def writable_keys(self, region_name: str) -> set[str]:
        """Return the set of keys a region is allowed to write."""
        return self._writers.get(region_name, set())

    @property
    def all_registered_keys(self) -> set[str]:
        """Return the union of all registered writer keys."""
        result: set[str] = set()
        for keys in self._writers.values():
            result |= keys
        return result
