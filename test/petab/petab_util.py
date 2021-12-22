import os

import git

repo_base = "doc/example/tmp/benchmark-models/"

if not os.path.exists(repo_base):
    git.Repo.clone_from(
        "git://github.com/Benchmarking-Initiative/Benchmark-Models-PEtab.git",
        repo_base,
        depth=1,
    )
g = git.Git(repo_base)

# update repo if online
try:
    g.pull()
except git.exc.GitCommandError:
    pass

# model folder base
folder_base = repo_base + "Benchmark-Models/"
