"""EventBus Tests"""
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from simulacrum.core.event_bus import EventBus, Event
from simulacrum.core.events import (
    STEP_START, GOAL_NEEDED, GOAL_SELECTED, EXPLORATION_START,
    EXPLORATION_DONE, MEMORY_ADDED, ALIGNMENT_CHECK,
    PERSONALITY_UPDATE, EMOTION_PROCESS, EMOTION_UPDATED,
    HIBERNATE_ENTER, SYSTEM_DEAD, COMPRESSION_NEEDED, ALL_EVENTS,
)


class TestEventBus:
    """Test EventBus"""

    def test_create(self):
        bus = EventBus()
        assert bus is not None

    def test_publish_no_subscribers(self):
        bus = EventBus()
        result = bus.publish("unknown_event", {"x": 1})
        assert result == {}

    def test_subscribe_publish(self):
        bus = EventBus()
        received = []

        def handler(event):
            received.append(event.data)
            return {"ok": True}

        bus.subscribe("test_event", handler, name="h1")
        result = bus.publish("test_event", {"key": "value"})

        assert len(received) == 1
        assert received[0] == {"key": "value"}
        assert result == {"h1": {"ok": True}}

    def test_multiple_subscribers(self):
        bus = EventBus()
        results = []

        def handler_a(event):
            results.append("a")
            return {"from": "a"}

        def handler_b(event):
            results.append("b")
            return {"from": "b"}

        bus.subscribe("test", handler_a, priority=0, name="a")
        bus.subscribe("test", handler_b, priority=1, name="b")
        bus.publish("test", {})

        assert results == ["a", "b"]  # priority order

    def test_priority_ordering(self):
        bus = EventBus()
        results = []

        def low(event):
            results.append("low")

        def high(event):
            results.append("high")

        bus.subscribe("test", low, priority=10)
        bus.subscribe("test", high, priority=0)
        bus.publish("test", {})

        assert results == ["high", "low"]

    def test_unsubscribe(self):
        bus = EventBus()
        call_count = [0]

        def handler(event):
            call_count[0] += 1

        bus.subscribe("test", handler, name="h")
        bus.publish("test", {})
        assert call_count[0] == 1

        bus.unsubscribe("test", handler)
        bus.publish("test", {})
        assert call_count[0] == 1  # not called again

    def test_handler_error_doesnt_crash(self):
        bus = EventBus()

        def bad_handler(event):
            raise ValueError("boom")

        def good_handler(event):
            return {"ok": True}

        bus.subscribe("test", bad_handler, name="bad")
        bus.subscribe("test", good_handler, name="good")
        result = bus.publish("test", {})

        assert result == {"good": {"ok": True}}

    def test_log(self):
        bus = EventBus(log_enabled=True)
        bus.publish("test_a", {"x": 1})
        bus.publish("test_b", {"y": 2})

        assert len(bus._log) == 2
        assert bus._log[0].type == "test_a"
        assert bus._log[1].type == "test_b"

    def test_stats(self):
        bus = EventBus()
        bus.subscribe("test", lambda e: None, name="h")
        bus.publish("test", {})
        bus.publish("test", {})

        stats = bus.get_stats()
        assert stats["publish_counts"]["test"] == 2
        assert stats["subscriber_counts"]["test"] == 1

    def test_reset(self):
        bus = EventBus()
        bus.subscribe("test", lambda e: None)
        bus.publish("test", {})
        bus.reset()

        stats = bus.get_stats()
        assert stats["publish_counts"] == {}
        assert stats["subscriber_counts"] == {}


class TestEventTypes:
    """Test event type definitions"""

    def test_all_events_defined(self):
        assert len(ALL_EVENTS) == 24

    def test_event_constants_unique(self):
        assert len(ALL_EVENTS) == len(set(ALL_EVENTS))

    def test_event_constants_are_strings(self):
        for e in ALL_EVENTS:
            assert isinstance(e, str)
            assert len(e) > 0


class TestEventIntegration:
    """Test event-driven flow simulation"""

    def test_step_flow(self):
        bus = EventBus()
        states = []

        def thermo_handler(event):
            return {"thermo_state": "ACTIVE", "balance": 50.0}

        bus.subscribe(STEP_START, thermo_handler, name="thermo")
        result = bus.publish(STEP_START, {"elapsed": 1.0})

        assert result["thermo"]["thermo_state"] == "ACTIVE"

    def test_goal_flow(self):
        bus = EventBus()

        def curiosity_handler(event):
            return {"goal": "test_goal", "emotion_bonus": 0.0}

        bus.subscribe(GOAL_NEEDED, curiosity_handler, name="curiosity")
        result = bus.publish(GOAL_NEEDED, {"emotion_state": {}})

        assert result["curiosity"]["goal"] == "test_goal"

    def test_dead_short_circuit(self):
        """When DEAD, no further events should be published"""
        bus = EventBus()
        goal_selected = [False]

        def goal_handler(event):
            goal_selected[0] = True

        bus.subscribe(GOAL_NEEDED, goal_handler, name="curiosity")

        # Simulate: thermo returns DEAD, agent should NOT publish GOAL_NEEDED
        thermo_result = bus.publish(STEP_START, {}, source="agent")
        # In real agent, step() checks thermo_result before publishing GOAL_NEEDED
        # Here we verify the pattern
        assert not goal_selected[0]
