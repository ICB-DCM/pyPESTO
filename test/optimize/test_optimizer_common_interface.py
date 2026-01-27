"""Test unified interface for optimizer limits (time, iterations, evaluations)."""

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


class TestOptimizerMaxiterInterface:
    """Test the unified maxiter interface for optimizers."""

    def test_scipy_optimizer_support(self):
        """Test ScipyOptimizer iteration limit support."""
        optimizer = optimize.ScipyOptimizer()
        assert optimizer.supports_maxiter() is True

        optimizer.set_maxiter(100)
        assert optimizer.options["maxiter"] == 100

    def test_ipopt_optimizer_support(self):
        """Test IpoptOptimizer iteration limit support."""
        optimizer = optimize.IpoptOptimizer()
        assert optimizer.supports_maxiter() is True

        optimizer.set_maxiter(100)
        assert optimizer.options["max_iter"] == 100

    def test_dlib_optimizer_support(self):
        """Test DlibOptimizer iteration limit support."""
        optimizer = optimize.DlibOptimizer()
        assert optimizer.supports_maxiter() is True

        optimizer.set_maxiter(500)
        assert optimizer.options["maxiter"] == 500

    def test_pyswarm_optimizer_support(self):
        """Test PyswarmOptimizer iteration limit support."""
        optimizer = optimize.PyswarmOptimizer()
        assert optimizer.supports_maxiter() is True

        optimizer.set_maxiter(150)
        assert optimizer.options["maxiter"] == 150

    def test_cma_optimizer_support(self):
        """Test CmaOptimizer iteration limit support."""
        optimizer = optimize.CmaOptimizer()
        assert optimizer.supports_maxiter() is True

        optimizer.set_maxiter(5000)
        assert optimizer.options["maxiter"] == 5000

    def test_scipy_de_optimizer_support(self):
        """Test ScipyDifferentialEvolutionOptimizer iteration limit support."""
        optimizer = optimize.ScipyDifferentialEvolutionOptimizer()
        assert optimizer.supports_maxiter() is True

        optimizer.set_maxiter(200)
        assert optimizer.options["maxiter"] == 200

    def test_pyswarms_optimizer_support(self):
        """Test PyswarmsOptimizer iteration limit support."""
        optimizer = optimize.PyswarmsOptimizer()
        assert optimizer.supports_maxiter() is True

        optimizer.set_maxiter(800)
        assert optimizer.options["maxiter"] == 800

    def test_nlopt_optimizer_no_support(self):
        """Test that NLoptOptimizer does not support iteration limits."""
        optimizer = optimize.NLoptOptimizer()
        # NLopt converts maxiter to maxeval, so it doesn't support maxiter directly
        assert optimizer.supports_maxiter() is False

        with pytest.raises(NotImplementedError):
            optimizer.set_maxiter(100)

    def test_fides_optimizer_support(self):
        """Test FidesOptimizer iteration limit support."""
        optimizer = optimize.FidesOptimizer()
        assert optimizer.supports_maxiter() is True

        optimizer.set_maxiter(100)

        from fides.constants import Options as FidesOptions

        assert FidesOptions.MAXITER in optimizer.options
        assert optimizer.options[FidesOptions.MAXITER] == 100

    def test_ess_optimizer_support(self):
        """Test ESSOptimizer iteration limit support."""
        from pypesto.optimize.ess import ESSOptimizer

        optimizer = ESSOptimizer(max_walltime_s=100.0)
        assert optimizer.supports_maxiter() is True

        optimizer.set_maxiter(50)
        assert optimizer.max_iter == 50


class TestOptimizerMaxevalInterface:
    """Test the unified maxeval interface for optimizers."""

    def test_nlopt_optimizer_support(self):
        """Test NLoptOptimizer evaluation limit support."""
        optimizer = optimize.NLoptOptimizer()
        assert optimizer.supports_maxeval() is True

        optimizer.set_maxeval(1000)
        assert optimizer.options["maxeval"] == 1000

    def test_cma_optimizer_support(self):
        """Test CmaOptimizer evaluation limit support."""
        optimizer = optimize.CmaOptimizer()
        assert optimizer.supports_maxeval() is True

        optimizer.set_maxeval(2000)
        assert optimizer.options["maxfevals"] == 2000

    def test_ess_optimizer_support(self):
        """Test ESSOptimizer evaluation limit support."""
        from pypesto.optimize.ess import ESSOptimizer

        optimizer = ESSOptimizer(max_walltime_s=100.0)
        assert optimizer.supports_maxeval() is True

        optimizer.set_maxeval(300)
        assert optimizer.max_eval == 300

    def test_ipopt_optimizer_no_support(self):
        """Test that IpoptOptimizer does not support evaluation limits."""
        optimizer = optimize.IpoptOptimizer()
        # Ipopt only has max_iter, not maxeval
        assert optimizer.supports_maxeval() is False

        with pytest.raises(NotImplementedError):
            optimizer.set_maxeval(100)


class TestOptimizerTolInterface:
    """Test the unified tolerance interface for optimizers."""

    def test_scipy_optimizer_support(self):
        """Test ScipyOptimizer tolerance support."""
        optimizer = optimize.ScipyOptimizer()
        assert optimizer.supports_tol() is True

        optimizer.set_tol(1e-6)
        assert optimizer.tol == 1e-6

        # Test updating existing value
        optimizer.set_tol(1e-8)
        assert optimizer.tol == 1e-8

    def test_ipopt_optimizer_support(self):
        """Test IpoptOptimizer tolerance support."""
        optimizer = optimize.IpoptOptimizer()
        assert optimizer.supports_tol() is True

        optimizer.set_tol(1e-7)
        assert optimizer.options["tol"] == 1e-7

    def test_nlopt_optimizer_support(self):
        """Test NLoptOptimizer tolerance support."""
        optimizer = optimize.NLoptOptimizer()
        assert optimizer.supports_tol() is True

        optimizer.set_tol(1e-5)
        assert optimizer.options["ftol_abs"] == 1e-5

    def test_fides_optimizer_support(self):
        """Test FidesOptimizer tolerance support."""
        optimizer = optimize.FidesOptimizer()
        assert optimizer.supports_tol() is True

        optimizer.set_tol(1e-6)

        from fides.constants import Options as FidesOptions

        assert FidesOptions.FATOL in optimizer.options
        assert optimizer.options[FidesOptions.FATOL] == 1e-6

        # Test updating existing value
        optimizer.set_tol(1e-9)
        assert optimizer.options[FidesOptions.FATOL] == 1e-9

    def test_cma_optimizer_support(self):
        """Test CmaOptimizer tolerance support."""
        optimizer = optimize.CmaOptimizer()
        assert optimizer.supports_tol() is True

        optimizer.set_tol(1e-4)
        assert optimizer.options["tolfun"] == 1e-4

    def test_scipy_de_optimizer_support(self):
        """Test ScipyDifferentialEvolutionOptimizer tolerance support."""
        optimizer = optimize.ScipyDifferentialEvolutionOptimizer()
        assert optimizer.supports_tol() is True

        optimizer.set_tol(1e-5)
        assert optimizer.options["atol"] == 1e-5

    def test_pyswarm_optimizer_support(self):
        """Test PyswarmOptimizer tolerance support."""
        optimizer = optimize.PyswarmOptimizer()
        assert optimizer.supports_tol() is True

        optimizer.set_tol(1e-7)
        assert optimizer.options["minfunc"] == 1e-7

    def test_dlib_optimizer_no_support(self):
        """Test that DlibOptimizer does not support tolerance."""
        optimizer = optimize.DlibOptimizer()
        assert optimizer.supports_tol() is False

        with pytest.raises(NotImplementedError):
            optimizer.set_tol(1e-6)

    def test_pyswarms_optimizer_no_support(self):
        """Test that PyswarmsOptimizer does not support tolerance."""
        optimizer = optimize.PyswarmsOptimizer()
        assert optimizer.supports_tol() is False

        with pytest.raises(NotImplementedError):
            optimizer.set_tol(1e-6)

    def test_tolerance_validation(self):
        """Test that invalid tolerance values are rejected."""
        optimizer = optimize.ScipyOptimizer()

        # Test that positive values work
        optimizer.set_tol(1e-6)
        assert optimizer.tol == 1e-6

        # Test that zero is allowed (optimize as accurately as possible)
        optimizer.set_tol(0.0)
        assert optimizer.tol == 0.0

        # Test that negative values are rejected
        with pytest.raises(ValueError, match="must be non-negative"):
            optimizer.set_tol(-1e-6)

    def test_tolerance_validation_options_based(self):
        """Test tolerance validation for options-based optimizers."""
        optimizer = optimize.IpoptOptimizer()

        # Test that positive values work
        optimizer.set_tol(1e-7)
        assert optimizer.options["tol"] == 1e-7

        # Test that zero is allowed
        optimizer.set_tol(0.0)
        assert optimizer.options["tol"] == 0.0

        # Test that negative values are rejected
        with pytest.raises(ValueError, match="must be positive"):
            optimizer.set_tol(-1e-7)
