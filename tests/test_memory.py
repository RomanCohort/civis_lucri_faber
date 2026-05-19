"""Memory Tests"""
import pytest
import numpy as np
from simulacrum.core.hippocampus import Hippocampus
from simulacrum.utils.memory import KnowledgeMemory


class TestHippocampus:
    """Test hippocampus memory module"""

    def test_init(self):
        """Test initialization"""
        mem = Hippocampus(input_dim=64, encoding_dim=128)
        assert mem is not None

    def test_store(self):
        """Test storing (encode_memory)"""
        mem = Hippocampus(input_dim=64, encoding_dim=64)
        mem.encode_memory(
            state=np.random.randn(64),
            action="test_action",
            reward=1.0,
        )

    def test_retrieve(self):
        """Test retrieving"""
        mem = Hippocampus(input_dim=64, encoding_dim=64)
        # Add some memories first
        for i in range(10):
            mem.encode_memory(np.random.randn(64), str(i), 1.0)
        # Retrieve
        results = mem.retrieve(query=np.random.randn(64), top_k=5)
        assert len(results) <= 5


class TestKnowledgeMemory:
    """Test knowledge memory"""

    def test_init(self):
        """Test initialization"""
        mem = KnowledgeMemory(max_size=100)
        assert mem is not None

    def test_add_memory(self):
        """Test adding memory"""
        import tempfile
        import os
        # Use a temp file so we start with a clean memory store
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tmp_path = f.name
        try:
            mem = KnowledgeMemory(max_size=50, memory_path=tmp_path)
            mem.add_memory("test content", importance=0.5)
            assert len(mem.memories) == 1
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_add_experience(self):
        """Test adding experience"""
        mem = KnowledgeMemory(max_size=50)
        mem.add_experience(np.random.randn(64), "0", 1.0, np.random.randn(64))
        assert len(mem.experiences) == 1

    def test_get_recent_memories(self):
        """Test getting recent memories"""
        mem = KnowledgeMemory(max_size=50)
        for i in range(10):
            mem.add_memory(f"memory {i}", importance=0.5)
        recent = mem.get_recent_memories(n=3)
        assert len(recent) <= 3
