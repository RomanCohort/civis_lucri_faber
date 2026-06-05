"""Agent sub-modules for the Simulacrum orchestrator.

These mixins decompose the monolithic agent.py into logical groups:
- ChatMixin: chat interface, LLM interaction, cognitive gating
- StateMixin: state aggregation, statistics, save/load
- VocalizationMixin: vocalization input prep, emotion vector, censor results

Usage:
    from core.agent import ChatMixin, StateMixin, VocalizationMixin

    class Simulacrum(ChatMixin, StateMixin, VocalizationMixin):
        ...
"""

from core.agent_mixins.chat_mixin import ChatMixin
from core.agent_mixins.state_mixin import StateMixin
from core.agent_mixins.vocalization_mixin import VocalizationMixin

__all__ = [
    "ChatMixin",
    "StateMixin",
    "VocalizationMixin",
]