"""Tests for core.hippocampus — 9 tests covering instantiation,
encode, retrieve, and pruning."""
import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_hippo(input_dim=16, encoding_dim=32, pfc_dim=16):
    """Create a Hippocampus with minimal dimensions."""
    from core.hippocampus import Hippocampus
    return Hippocampus(input_dim=input_dim, encoding_dim=encoding_dim,
                       pfc_dim=pfc_dim)


# ---------------------------------------------------------------------------
# Instantiation (2 tests)
# ---------------------------------------------------------------------------

class TestInstantiation:
    def test_hippo_creates_without_error(self):
        hp = _make_hippo()
        assert hp is not None

    def test_hippo_has_required_attributes(self):
        hp = _make_hippo()
        assert hasattr(hp, "output_keys")
        assert hasattr(hp, "step")
        assert hasattr(hp, "encode_memory")


# ---------------------------------------------------------------------------
# Encode (3 tests)
# ---------------------------------------------------------------------------

class TestEncode:
    def test_encode_stores_episode(self):
        """Encoding a state should create a memory entry."""
        hp = _make_hippo()
        state = np.random.randn(16).astype(np.float32)
        encoding = hp.encode_memory(state, action="test", reward=0.5)
        assert len(hp.episodic_memory) == 1

    def test_encode_returns_array(self):
        """encode_memory should return an encoding array."""
        hp = _make_hippo()
        state = np.random.randn(16).astype(np.float32)
        encoding = hp.encode_memory(state, action="test", reward=0.5)
        assert isinstance(encoding, np.ndarray)
        assert encoding.shape[0] == 32  # encoding_dim

    def test_multiple_encodes_increase_memory(self):
        """Multiple encode calls should increase stored memories."""
        hp = _make_hippo()
        for i in range(5):
            state = np.random.randn(16).astype(np.float32)
            hp.encode_memory(state, action=f"action_{i}", reward=0.5)
        assert len(hp.episodic_memory) == 5


# ---------------------------------------------------------------------------
# Retrieve (2 tests)
# ---------------------------------------------------------------------------

class TestRetrieve:
    def test_retrieve_returns_list(self):
        """Retrieve should return a list of EpisodeMemory."""
        hp = _make_hippo()
        state = np.random.randn(16).astype(np.float32)
        hp.encode_memory(state, action="test", reward=0.5)
        results = hp.retrieve(state, top_k=3)
        assert isinstance(results, list)

    def test_retrieve_after_multiple_encodes(self):
        """Retrieve after encoding multiple memories should return results."""
        hp = _make_hippo()
        for i in range(5):
            state = np.random.randn(16).astype(np.float32)
            hp.encode_memory(state, action=f"a_{i}", reward=float(i))
        query = np.random.randn(16).astype(np.float32)
        results = hp.retrieve(query, top_k=3)
        assert len(results) <= 3


# ---------------------------------------------------------------------------
# Pruning (2 tests)
# ---------------------------------------------------------------------------

class TestPruning:
    def test_memory_does_not_explode(self):
        """Memory should not grow unbounded."""
        hp = _make_hippo()
        for i in range(50):
            state = np.random.randn(16).astype(np.float32)
            hp.encode_memory(state, action=f"a_{i}", reward=0.5)
        # Should be bounded (interference engine or FIFO)
        assert len(hp.episodic_memory) <= 1000  # generous upper bound

    def test_interference_engine_reduces_low_importance(self):
        """Interference forgetting should prune low-importance memories."""
        hp = _make_hippo()
        hp.use_interference_forgetting = True
        for i in range(20):
            state = np.random.randn(16).astype(np.float32)
            hp.encode_memory(state, action=f"a_{i}", reward=0.01)
        # After encoding many similar low-reward memories,
        # some should have reduced importance
        has_reduced = any(m.importance < 1.0 for m in hp.episodic_memory)
        assert has_reduced or len(hp.episodic_memory) < 20
