"""Brain Region Tests"""
from core.basal_ganglia import BasalGanglia
from core.cerebral_cortex import VisualCortex
from core.hippocampus import Hippocampus
from core.limbic import LimbicSystem
from core.prefrontal_cortex import PrefrontalCortex


class TestVisualCortex:
    """Test visual cortex (cerebral_cortex module)"""

    def test_init(self):
        """Test initialization"""
        cortex = VisualCortex(input_channels=3, embed_dim=64)
        assert cortex is not None


class TestPrefrontalCortex:
    """Test prefrontal cortex"""

    def test_init(self):
        """Test initialization"""
        pfc = PrefrontalCortex(input_dim=64, hidden_dim=128)
        assert pfc is not None


class TestLimbicSystem:
    """Test limbic system"""

    def test_init(self):
        """Test initialization"""
        limbic = LimbicSystem()
        assert limbic is not None


class TestBasalGanglia:
    """Test basal ganglia"""

    def test_init(self):
        """Test initialization"""
        bg = BasalGanglia(state_dim=64, n_actions=4)
        assert bg is not None


class TestHippocampus:
    """Test hippocampus"""

    def test_init(self):
        """Test initialization"""
        hippo = Hippocampus(input_dim=64, encoding_dim=128)
        assert hippo is not None
