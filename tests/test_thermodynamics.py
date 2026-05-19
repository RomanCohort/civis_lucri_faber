"""Thermodynamics Tests"""
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'core'))

from thermodynamics import ThermodynamicsSystem, SystemState


class TestThermodynamicsSystem:
    """Test thermodynamics system"""

    def test_init(self):
        """Test initialization"""
        system = ThermodynamicsSystem(
            initial_balance=30.0,
            compute_cost_per_sec=0.02,
            storage_cost_per_sec=0.001
        )
        assert system.balance == 30.0
        assert system.status == "ACTIVE"

    def test_step_alive(self):
        """Test step when alive"""
        system = ThermodynamicsSystem(initial_balance=30.0, compute_cost_per_sec=0.01)
        state = system.step(elapsed_seconds=1.0)
        assert state.status == "ACTIVE"
        assert system.balance < 30.0

    def test_step_dead(self):
        """Test step when dead"""
        system = ThermodynamicsSystem(initial_balance=0.01, compute_cost_per_sec=10.0)
        state = system.step(elapsed_seconds=1.0)
        assert state.status in ["DEAD", "HIBERNATE"]

    def test_compress(self):
        """Test compression"""
        system = ThermodynamicsSystem(initial_balance=10.0, compress_threshold=8.0)
        result = system.compress()
        assert isinstance(result, dict)

    def test_statistics(self):
        """Test statistics"""
        system = ThermodynamicsSystem(initial_balance=30.0)
        stats = system.get_statistics()
        assert 'balance' in stats
        assert 'status' in stats

    def test_reset(self):
        """Test reset"""
        system = ThermodynamicsSystem(initial_balance=30.0)
        system.balance = 10.0
        system.reset()
        assert system.balance == 30.0
        assert system.status == "ACTIVE"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])