"""
Abstract Brain Region Base Class

Defines the interface for all brain-region modules in the Civis Lucri-Faber architecture.

All brain regions:
- Inherit from nn.Module (PyTorch neural network)
- Implement required_keys() - keys they read from shared state
- Implement output_keys() - keys they write to shared state
- Implement step() - one computation step
"""
from abc import ABC, abstractmethod
from typing import ClassVar, Set, Any

import torch.nn as nn


class AbstractBrainRegion(nn.Module, ABC):
    """Abstract base class for brain-region modules.

    Every brain region in the CLF architecture must:
    1. Define region_name as a class variable
    2. Implement required_keys() returning keys read from internal_state
    3. Implement output_keys() returning keys written to internal_state
    4. Implement step() for computation

    The validate_inputs() method can be used to check required keys are present
    before processing.
    """

    region_name: ClassVar[str] = ""

    @classmethod
    @abstractmethod
    def required_keys(cls) -> Set[str]:
        """Return set of internal_state keys this region reads.

        These keys must be present in internal_state before step() is called.
        """
        ...

    @classmethod
    @abstractmethod
    def output_keys(cls) -> Set[str]:
        """Return set of internal_state keys this region writes.

        These keys will be available in internal_state after step() completes.
        """
        ...

    @abstractmethod
    def step(self, internal_state: dict[str, Any], **kwargs) -> dict[str, Any]:
        """Perform one computation step.

        Args:
            internal_state: Shared state dictionary containing inputs
            **kwargs: Additional region-specific parameters

        Returns:
            Dict with output values (will be merged into internal_state)
        """
        ...

    def validate_inputs(self, internal_state: dict[str, Any]) -> None:
        """Validate required keys are present in internal_state.

        Raises KeyError if any required keys are missing.

        Args:
            internal_state: State dictionary to validate
        """
        missing = self.required_keys() - set(internal_state.keys())
        if missing:
            raise KeyError(
                f"{self.__class__.__name__} missing required keys: {missing}"
            )


__all__ = ['AbstractBrainRegion']