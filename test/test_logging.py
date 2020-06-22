import logging
import os
import pypesto
import pypesto.optimize


def test_optimize():
    # logging
    pypesto.logging.log_to_console(logging.WARN)
    filename = ".test_logging.tmp"
    pypesto.logging.log_to_file(logging.DEBUG, filename)
    logger = logging.getLogger('pypesto')
    if os.path.exists(filename):
        os.remove(filename)
    fh = logging.FileHandler(filename)
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    logger.info("start test")

    # problem definition
    def fun(_):
        raise Exception("This function cannot be called.")

    objective = pypesto.Objective(fun=fun)
    problem = pypesto.Problem(objective, -1, 1)

    optimizer = pypesto.optimize.ScipyOptimizer()
    options = {'allow_failed_starts': True}

    # optimization
    pypesto.optimize.minimize(problem, optimizer, 5, options=options)

    # assert logging worked
    assert os.path.exists(filename)
    f = open(filename, 'rb')
    content = str(f.read())
    f.close()

    # tidy up
    os.remove(filename)

    # check if error message got inserted
    assert "fail" in content
