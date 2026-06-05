"""EventBus integration tests — verify event flow between brain regions."""
import pytest
import torch

from core.events import (
    EventBus,
    Event,
    ALL_EVENTS,
    EMOTION_PROCESS,
    EMOTION_UPDATED,
    HIBERNATE_ENTER,
    HIBERNATE_EXIT,
    METABOLIC_BUDGET_LOW,
    SLEEP_CYCLE,
    REWARD_SIGNAL,
    ACTION_SELECTED,
    CURIOSITY_TRIGGER,
    NEUROTRANSMITTER_UPDATE,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def bus():
    return EventBus()


@pytest.fixture
def received():
    """Shared list to capture events."""
    return []


# ---------------------------------------------------------------------------
# 1. Multi-subscriber priority ordering
# ---------------------------------------------------------------------------

class TestPriorityOrdering:
    def test_higher_priority_called_first(self, bus, received):
        """Subscribers with higher priority (lower number) fire first."""
        bus.subscribe("TEST_EVENT", lambda e: received.append("low"), priority=10, name="low")
        bus.subscribe("TEST_EVENT", lambda e: received.append("high"), priority=0, name="high")
        bus.publish("TEST_EVENT", {"v": 1}, source="test")
        assert received == ["high", "low"]

    def test_same_priority_fifo(self, bus, received):
        """Same-priority subscribers fire in FIFO order."""
        bus.subscribe("TEST_EVENT", lambda e: received.append("first"), priority=5, name="first")
        bus.subscribe("TEST_EVENT", lambda e: received.append("second"), priority=5, name="second")
        bus.publish("TEST_EVENT", {"v": 1}, source="test")
        assert received == ["first", "second"]


# ---------------------------------------------------------------------------
# 2. Event data propagation
# ---------------------------------------------------------------------------

class TestDataPropagation:
    def test_event_carries_data(self, bus, received):
        """Published event data reaches subscriber."""
        def handler(event):
            received.append(event.data["value"])

        bus.subscribe("DATA_TEST", handler, priority=0, name="data")
        bus.publish("DATA_TEST", {"value": 42}, source="test")
        assert received == [42]

    def test_event_source_tracking(self, bus, received):
        """Event source is preserved through the bus."""
        def handler(event):
            received.append(event.source)

        bus.subscribe("SRC_TEST", handler, priority=0, name="src")
        bus.publish("SRC_TEST", {}, source="brainstem")
        assert received == ["brainstem"]


# ---------------------------------------------------------------------------
# 3. Cross-region event chain
# ---------------------------------------------------------------------------

class TestCrossRegionChain:
    def test_emotion_process_publishes_updated(self, bus):
        """EMOTION_PROCESS → advanced_emotion → EMOTION_UPDATED chain."""
        updated_received = []

        # Simulate what advanced_emotion does
        def on_emotion_process(event):
            bus.publish(EMOTION_UPDATED, {"emotion_state": {"current_emotion": "joy"}}, source="advanced_emotion")

        bus.subscribe(EMOTION_PROCESS, on_emotion_process, priority=0, name="emotion_handler")
        bus.subscribe(EMOTION_UPDATED, lambda e: updated_received.append(e.data), priority=0, name="downstream")

        bus.publish(EMOTION_PROCESS, {"user_input": "hello"}, source="agent")
        assert len(updated_received) == 1
        assert updated_received[0]["emotion_state"]["current_emotion"] == "joy"

    def test_hibernate_triggers_consolidation(self, bus, received):
        """HIBERNATE_ENTER triggers consolidation handler."""
        bus.subscribe(HIBERNATE_ENTER, lambda e: received.append("consolidated"), priority=0, name="consolidation")
        bus.publish(HIBERNATE_ENTER, {}, source="sleep_system")
        assert "consolidated" in received


# ---------------------------------------------------------------------------
# 4. Unsubscribe and isolation
# ---------------------------------------------------------------------------

class TestUnsubscribe:
    def test_unsubscribe_stops_delivery(self, bus, received):
        """Unsubscribed handler no longer receives events."""
        bus.subscribe("UNSUB_TEST", lambda e: received.append("called"), priority=0, name="unsub_me")
        bus.publish("UNSUB_TEST", {}, source="test")
        assert len(received) == 1

        bus.unsubscribe("UNSUB_TEST", "unsub_me")
        bus.publish("UNSUB_TEST", {}, source="test")
        assert len(received) == 1  # Not called again

    def test_different_events_isolated(self, bus, received):
        """Subscribers for event A don't receive event B."""
        bus.subscribe("EVENT_A", lambda e: received.append("A"), priority=0, name="a")
        bus.subscribe("EVENT_B", lambda e: received.append("B"), priority=0, name="b")
        bus.publish("EVENT_A", {}, source="test")
        assert received == ["A"]


# ---------------------------------------------------------------------------
# 5. Error resilience
# ---------------------------------------------------------------------------

class TestErrorResilience:
    def test_failing_handler_doesnt_block_others(self, bus, received):
        """One handler raising an exception doesn't prevent other handlers."""
        def bad_handler(event):
            raise RuntimeError("boom")

        def good_handler(event):
            received.append("good")

        bus.subscribe("ERR_TEST", bad_handler, priority=0, name="bad")
        bus.subscribe("ERR_TEST", good_handler, priority=1, name="good")
        # Should not raise
        bus.publish("ERR_TEST", {}, source="test")
        assert "good" in received


# ---------------------------------------------------------------------------
# 6. ALL_EVENTS registry completeness
# ---------------------------------------------------------------------------

class TestEventRegistry:
    def test_all_events_are_unique(self):
        """No duplicate event names in ALL_EVENTS."""
        assert len(ALL_EVENTS) == len(set(ALL_EVENTS))

    def test_core_events_present(self):
        """Key brain-region events are registered."""
        core = [EMOTION_PROCESS, EMOTION_UPDATED, HIBERNATE_ENTER, HIBERNATE_EXIT,
                METABOLIC_BUDGET_LOW, SLEEP_CYCLE, REWARD_SIGNAL, ACTION_SELECTED,
                CURIOSITY_TRIGGER, NEUROTRANSMITTER_UPDATE]
        for ev in core:
            assert ev in ALL_EVENTS, f"{ev} missing from ALL_EVENTS"


# ---------------------------------------------------------------------------
# 7. Event object structure
# ---------------------------------------------------------------------------

class TestEventObject:
    def test_event_has_required_fields(self):
        """Event dataclass has type, data, source."""
        ev = Event(type="TEST", data={"x": 1}, source="unit_test")
        assert ev.type == "TEST"
        assert ev.data == {"x": 1}
        assert ev.source == "unit_test"

    def test_event_default_data(self):
        """Event defaults to empty dict for data."""
        ev = Event(type="TEST", source="unit_test")
        assert ev.data == {}
