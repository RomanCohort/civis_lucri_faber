"""Self Alignment Tests"""
import os as _os
import sys

import pytest

sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), 'core'))

from unittest.mock import Mock


class TestSelfAlignmentModule:
    """Test self alignment module"""

    def test_init_without_api(self):
        """Test initialization without API"""
        mock_client = Mock()
        mock_client.generate = Mock(return_value="aligned")

        from self_alignment import SelfAlignmentModule

        module = SelfAlignmentModule(
            api_client=mock_client,
            check_interval=5,
            log_path="test_alignment.json"
        )
        assert module is not None

    def test_get_alignment_score(self):
        """Test getting alignment score"""
        mock_client = Mock()
        mock_client.generate = Mock(return_value="aligned")

        from self_alignment import SelfAlignmentModule

        module = SelfAlignmentModule(api_client=mock_client, check_interval=5)
        score = module.get_alignment_score()
        assert isinstance(score, float)

    def test_statistics(self):
        """Test statistics"""
        mock_client = Mock()

        from self_alignment import SelfAlignmentModule

        module = SelfAlignmentModule(api_client=mock_client)
        stats = module.get_statistics()
        assert 'total_reflections' in stats or 'avg_alignment' in stats


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
