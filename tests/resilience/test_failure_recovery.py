import pytest

from app.resilience.circuit_breaker import CircuitBreaker, CircuitBreakerOpenError, CircuitState
from app.resilience.retry import vector_db_retry


def test_retry_recovers_after_transient_failures_then_succeeds():
    """Simulates the vector DB raising ConnectionError twice, then succeeding
    on the 3rd call — the pipeline should still get the correct result."""
    call_count = {"n": 0}

    @vector_db_retry
    def flaky_vector_db_call():
        call_count["n"] += 1
        if call_count["n"] < 3:
            raise ConnectionError("vector db unreachable")
        return {"matches": ["result"]}

    result = flaky_vector_db_call()

    assert result == {"matches": ["result"]}
    assert call_count["n"] == 3  # 2 failed attempts + 1 successful attempt


def test_retry_gives_up_after_max_attempts_and_reraises():
    call_count = {"n": 0}

    @vector_db_retry
    def always_fails():
        call_count["n"] += 1
        raise ConnectionError("vector db permanently down")

    with pytest.raises(ConnectionError):
        always_fails()

    assert call_count["n"] == 3  # stop_after_attempt(3), no more, no less


def test_retry_does_not_retry_non_retryable_exceptions():
    call_count = {"n": 0}

    @vector_db_retry
    def raises_value_error():
        call_count["n"] += 1
        raise ValueError("not a transient failure")

    with pytest.raises(ValueError):
        raises_value_error()

    assert call_count["n"] == 1  # ValueError isn't in the retry allowlist


def test_circuit_breaker_opens_after_consecutive_failures_and_fails_fast():
    breaker = CircuitBreaker(failure_threshold=3, cooldown_seconds=30)
    call_count = {"n": 0}

    def failing_call():
        call_count["n"] += 1
        raise ConnectionError("vector db down")

    for _ in range(3):
        with pytest.raises(ConnectionError):
            breaker.call(failing_call)

    assert breaker.state == CircuitState.OPEN
    assert call_count["n"] == 3

    # circuit is open: further calls fail fast without invoking the function
    with pytest.raises(CircuitBreakerOpenError):
        breaker.call(failing_call)
    assert call_count["n"] == 3  # not incremented — call never reached the function


def test_circuit_breaker_closes_again_after_successful_call():
    breaker = CircuitBreaker(failure_threshold=2, cooldown_seconds=30)

    with pytest.raises(ConnectionError):
        breaker.call(lambda: (_ for _ in ()).throw(ConnectionError()))

    result = breaker.call(lambda: "recovered")

    assert result == "recovered"
    assert breaker.state == CircuitState.CLOSED
