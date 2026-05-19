"""Metabolic Budget Tests"""
import pytest
import torch
from civis_lucri_faber.core.metabolic_budget import (
    MetabolicBudget, MetabolicCostCalculator, PeriodicStarvation
)


class TestMetabolicCostCalculator:
    """Test metabolic cost calculator"""

    def test_init(self):
        """Test initialization"""
        calc = MetabolicCostCalculator(resource_budget=0.3)
        assert calc.budget == 0.3

    def test_forward(self):
        """Test forward pass"""
        calc = MetabolicCostCalculator(resource_budget=0.3)
        hidden = torch.randn(2, 10, 32)
        cost, detail = calc(hidden)
        assert isinstance(cost.item(), float)

    def test_activation_rate(self):
        """Test activation rate calculation"""
        calc = MetabolicCostCalculator(resource_budget=0.3)
        # All active
        hidden = torch.ones(2, 10, 32)
        cost, detail = calc(hidden)
        assert cost > 0


class TestPeriodicStarvation:
    """Test periodic starvation"""

    def test_init(self):
        """Test initialization"""
        starvation = PeriodicStarvation(starvation_prob=0.15, cycle_steps=500)
        assert starvation.cycle_steps == 500

    def test_get_gate_mask(self):
        """Test gate mask"""
        starvation = PeriodicStarvation()
        importance = {"layer0": torch.rand(32)}
        masks, triggered = starvation.get_gate_mask(importance)
        assert isinstance(masks, dict)
        assert "layer0" in masks
        assert isinstance(triggered, bool)


class TestMetabolicBudget:
    """Test full metabolic budget system"""

    def test_init(self):
        """Test initialization"""
        budget = MetabolicBudget(resource_budget=0.3)
        assert budget is not None

    def test_compute_loss(self):
        """Test loss computation"""
        budget = MetabolicBudget(resource_budget=0.3)
        task_loss = torch.tensor(1.0)
        hidden = torch.randn(2, 10, 32)
        total_loss, details = budget.compute_loss(task_loss, hidden, return_detail=True)
        assert isinstance(total_loss.item(), float)

    def test_get_state(self):
        """Test getting state"""
        budget = MetabolicBudget(resource_budget=0.3)
        state = budget.get_state()
        assert hasattr(state, 'active_ratio') or isinstance(state, dict)