import logging
import time
from enum import Enum
from typing import Callable, TypeVar

logger = logging.getLogger("app.resilience")

T = TypeVar("T")


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreakerOpenError(Exception):
    pass


class CircuitBreaker:
    """Basic closed -> open -> half-open state machine.

    Opens after `failure_threshold` consecutive failures and fails fast
    (no network call) until `cooldown_seconds` has elapsed, then allows one
    trial call (half-open) to decide whether to close again or re-open.
    """

    def __init__(self, failure_threshold: int = 3, cooldown_seconds: int = 30):
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self._state = CircuitState.CLOSED
        self._consecutive_failures = 0
        self._opened_at: float | None = None

    @property
    def state(self) -> CircuitState:
        if self._state == CircuitState.OPEN and self._cooldown_elapsed():
            self._state = CircuitState.HALF_OPEN
        return self._state

    def _cooldown_elapsed(self) -> bool:
        return self._opened_at is not None and (time.monotonic() - self._opened_at) >= self.cooldown_seconds

    def call(self, func: Callable[[], T]) -> T:
        current_state = self.state
        if current_state == CircuitState.OPEN:
            logger.warning("Circuit breaker OPEN — failing fast without calling backend")
            raise CircuitBreakerOpenError(
                f"Circuit open; failing fast for cooldown ({self.cooldown_seconds}s)"
            )

        try:
            result = func()
        except Exception as exc:
            self._on_failure(exc)
            raise
        else:
            self._on_success()
            return result

    def _on_success(self) -> None:
        if self._consecutive_failures > 0:
            logger.info("Circuit breaker recovered — closing after successful call")
        self._consecutive_failures = 0
        self._state = CircuitState.CLOSED
        self._opened_at = None

    def _on_failure(self, exc: Exception) -> None:
        self._consecutive_failures += 1
        logger.warning(
            "Backend call failed (%d/%d consecutive failures): %s",
            self._consecutive_failures, self.failure_threshold, exc,
        )
        if self._consecutive_failures >= self.failure_threshold:
            self._state = CircuitState.OPEN
            self._opened_at = time.monotonic()
            logger.error(
                "Circuit breaker OPENING after %d consecutive failures — cooling down for %ds",
                self._consecutive_failures, self.cooldown_seconds,
            )
