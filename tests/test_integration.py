"""Integration Tests

Full system tests covering all 14+ mechanisms
"""
import pytest
import os
import sys
import numpy as np
import torch

# Add project to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from simulacrum.utils.config import load_config
from simulacrum.core.agent import Simulacrum


class TestAgentIntegration:
    """Integration tests for main agent"""

    @pytest.fixture
    def config(self):
        """Create test config"""
        return load_config(
            curiosity_alpha=0.4,
            curiosity_beta=0.3,
            curiosity_gamma=0.3,
            exploration_rate=0.2,
            intrinsic_motivation_lambda=0.5,
            meta_lr=0.01,
            alignment_check_interval=5,
            initial_balance=30.0,
            compute_cost_per_sec=0.02,
            storage_cost_per_sec=0.001,
            task_reward_min=0.1,
            task_reward_max=1.0,
            compress_threshold=8.0,
            openai_api_key="",
            anthropic_api_key="",
            model_name="gpt-4",
            max_history_size=500,
            device="cpu",
            seed=42
        )

    @pytest.fixture
    def agent(self, config):
        """Create test agent"""
        agent = Simulacrum(
            config=config,
            memory_path="test_memory.json",
            alignment_log_path="test_alignment.json",
            thermo_log_path="test_thermo.json",
            state_dim=4,
            n_actions=4
        )
        return agent

    def test_agent_init(self, agent):
        """Test agent initialization"""
        assert agent is not None
        assert agent.curiosity is not None
        assert agent.info_gain_calc is not None
        assert agent.thermo is not None

    def test_agent_step(self, agent):
        """Test single step"""
        state = agent.step()
        assert state is not None
        assert hasattr(state, 'step')
        assert hasattr(state, 'status')
        assert hasattr(state, 'balance')

    def test_agent_episodes(self, agent):
        """Test multiple episodes"""
        states = agent.run_episodes(n_episodes=5, verbose=False)
        assert len(states) > 0
        assert all(hasattr(s, 'step') for s in states)

    def test_agent_statistics(self, agent):
        """Test full statistics"""
        # Run a few steps
        agent.run_episodes(n_episodes=3, verbose=False)
        stats = agent.get_full_statistics()
        assert 'thermodynamics' in stats
        assert 'curiosity' in stats
        assert 'info_gain' in stats
        assert 'personality' in stats
        assert 'memory' in stats

    def test_personality_subsystems(self, agent):
        """Test all personality subsystems"""
        stats = agent.get_full_statistics()
        personality = stats['personality']

        # All 6 personality modules should be present
        assert 'identity' in personality
        assert 'relation' in personality
        assert 'attention' in personality
        assert 'motivation' in personality
        assert 'neuromodulation' in personality
        assert 'epigenetic' in personality

    def test_curiosity_mechanism(self, agent):
        """Test curiosity exploration"""
        stats = agent.get_full_statistics()
        curiosity_stats = stats['curiosity']

        assert 'total_goals' in curiosity_stats
        assert 'completed' in curiosity_stats

    def test_info_gain_mechanism(self, agent):
        """Test information gain"""
        stats = agent.get_full_statistics()
        ig_stats = stats['info_gain']

        assert 'buffer_size' in ig_stats or 'info_gain_avg' in ig_stats

    def test_thermodynamics_mechanism(self, agent):
        """Test thermodynamics (survival)"""
        stats = agent.get_full_statistics()
        thermo_stats = stats['thermodynamics']

        assert 'balance' in thermo_stats
        assert 'status' in thermo_stats

    def test_agent_reset(self, agent):
        """Test agent reset"""
        agent.run_episodes(n_episodes=2, verbose=False)
        agent.reset()
        assert agent.step_count == 0

    def test_agent_save_load(self, agent, tmp_path):
        """Test save and load"""
        # Run some steps
        agent.run_episodes(n_episodes=2, verbose=False)

        # Save
        save_path = tmp_path / "test_model.pt"
        agent.save(str(save_path))

        # Load creates new agent
        agent2 = Simulacrum(
            config=agent.config,
            state_dim=4,
            n_actions=4
        )
        agent2.load(str(save_path))

        # Verify basic state
        assert agent2.config.initial_balance == agent.config.initial_balance


class TestAllMechanismsPresent:
    """Test that all mechanisms are present in the system"""

    def test_core_modules_importable(self):
        """Test all core modules can be imported"""
        from simulacrum.core.curiosity import CuriosityEngine
        from simulacrum.core.information_gain import TrueInformationGainCalculator
        from simulacrum.core.meta_learning import FirstOrderMAML
        from simulacrum.core.self_alignment import SelfAlignmentModule
        from simulacrum.core.thermodynamics import ThermodynamicsSystem
        from simulacrum.core.policy_learning import SimpleQLearning
        from simulacrum.core.metabolic_budget import MetabolicBudget

        # All should be importable without error
        assert CuriosityEngine is not None
        assert TrueInformationGainCalculator is not None
        assert FirstOrderMAML is not None
        assert SelfAlignmentModule is not None
        assert ThermodynamicsSystem is not None
        assert SimpleQLearning is not None
        assert MetabolicBudget is not None

    def test_personality_modules_importable(self):
        """Test all personality modules can be imported"""
        from simulacrum.core.personality import (
            TripartiteCompetitiveEngine,
            StreamingIdentityCore,
            RelationalEmbedding,
            AttentionGating,
            MotivationSurvivalSystem,
            NeuromodulationSystem,
            EpigeneticLearner,
        )

        assert TripartiteCompetitiveEngine is not None
        assert StreamingIdentityCore is not None
        assert RelationalEmbedding is not None
        assert AttentionGating is not None
        assert MotivationSurvivalSystem is not None
        assert NeuromodulationSystem is not None
        assert EpigeneticLearner is not None


class TestAgentDeathAndSurvival:
    """Test agent survival mechanics"""

    def test_agent_death(self):
        """Test agent can die from energy depletion"""
        config = load_config(
            initial_balance=0.01,  # Very low balance
            compute_cost_per_sec=10.0,  # High cost
            task_reward_min=0.0,
            task_reward_max=0.0,
        )

        agent = Simulacrum(config=config)

        # Run until dead
        for _ in range(10):
            state = agent.step()
            if state.status == "DEAD":
                break

        # Verify death occurred
        final_state = agent.step()
        assert final_state.status in ["DEAD", "HIBERNATE"] or final_state.balance <= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])