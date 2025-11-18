"""Test unified interface for optimizer time limits (issue #1553)."""

import pytest

import pypesto.optimize as optimize


class TestOptimizerMaxtimeInterface:
    """Test the unified maxtime interface for optimizers."""

    def test_scipy_optimizer_no_support(self):
        """Test that ScipyOptimizer does not support time limits."""
        optimizer = optimize.ScipyOptimizer()
        assert optimizer.supports_maxtime() is False

        with pytest.raises(NotImplementedError):
            optimizer.set_maxtime(10.0)

    def test_ipopt_optimizer_support(self):
        """Test IpoptOptimizer time limit support."""
        optimizer = optimize.IpoptOptimizer()

        # Support depends on Ipopt version
        try:
            import cyipopt

            expected_support = cyipopt.IPOPT_VERSION >= (3, 14, 0)
        except (ImportError, AttributeError):
            expected_support = False

        assert optimizer.supports_maxtime() == expected_support

        if expected_support:
            optimizer.set_maxtime(10.0)
            assert optimizer.options["max_wall_time"] == 10.0
        else:
            with pytest.raises(NotImplementedError):
                optimizer.set_maxtime(10.0)

    def test_nlopt_optimizer_support(self):
        """Test NLoptOptimizer time limit support."""
        optimizer = optimize.NLoptOptimizer()
        assert optimizer.supports_maxtime() is True

        optimizer.set_maxtime(10.0)
        assert optimizer.options["maxtime"] == 10.0

    def test_fides_optimizer_support(self):
        """Test FidesOptimizer time limit support."""
        optimizer = optimize.FidesOptimizer()
        assert optimizer.supports_maxtime() is True

        optimizer.set_maxtime(10.0)

        from fides.constants import Options as FidesOptions

        assert FidesOptions.MAXTIME in optimizer.options
        assert optimizer.options[FidesOptions.MAXTIME] == 10.0

        # Test updating existing value
        optimizer.set_maxtime(15.5)
        assert optimizer.options[FidesOptions.MAXTIME] == 15.5

    def test_ess_optimizer_support(self):
        """Test ESSOptimizer time limit support."""
        from pypesto.optimize.ess import ESSOptimizer

        optimizer = ESSOptimizer(max_walltime_s=100.0)
        assert optimizer.supports_maxtime() is True

        optimizer.set_maxtime(10.0)
        assert optimizer.max_walltime_s == 10.0
