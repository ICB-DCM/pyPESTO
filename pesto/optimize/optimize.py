def optimize(problem, optimizer, result=None):

    optimizer_results = []

    for point in problem.starting_points:
        optimizer_results.append(optimizer.minimize(problem, point))

    fvals = []


    results = optimizer_results

    return results
