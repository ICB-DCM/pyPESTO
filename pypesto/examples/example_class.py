"""Define a simple example class for pypesto examples."""
import os
import tempfile
from abc import ABC, abstractmethod

import petab.v1 as petab
import requests

from ..objective import ObjectiveBase
from ..petab import PetabImporter
from ..problem import Problem


class PyPESTOExampleBase(ABC):
    """Abstract example class for pypesto examples.

    Examples provide a simple wrapper around generic and in documentation
    used examples. They provide a simple interface to load the objective and
    problem functions.
    """

    def __init__(
        self,
        name: str,
        description: str,
        detailed_description: str | None = None,
    ):
        """
        Initialize the example.

        Parameters
        ----------
        name:
            The name of the example.
        description:
            A short description of the example.
        detailed_description:
            A detailed description of the example.
        """
        self.name = name
        self.description = description
        self.detailed_description = detailed_description

    @property
    @abstractmethod
    def objective(self) -> ObjectiveBase:
        """Returns the objective function of this example."""
        pass

    @property
    @abstractmethod
    def problem(self) -> Problem:
        """Returns the problem of this example."""
        pass


class PyPESTOExamplePEtab(PyPESTOExampleBase):
    """A PEtab example class for pypesto examples."""

    def __init__(
        self,
        name: str,
        description: str,
        github_repo: str,
        filenames: list[str],
        detailed_description: str | None = None,
        hierarchical: bool = False,
    ):
        """
        Initialize the example.

        Parameters
        ----------
        name:
            The name of the example.
        description:
            A short description of the example.
        github_repo:
            The github repository to download the example from.
        filenames:
            The filenames to download.
        detailed_description:
            A detailed description of the example.
        hierarchical:
            Whether the example is hierarchical problem.
            Needs to be set for problem creation.
        """
        super().__init__(name, description, detailed_description)
        self.hierarchical = hierarchical
        self.github_repo = github_repo
        self.filenames = filenames
        self.petab_yaml = next(
            (filename for filename in filenames if filename.endswith(".yaml")),
            None,
        )

        self._importer = None
        self._petab_problem = None
        self._objective = None
        self._problem = None

    @property
    def petab_problem(self) -> petab.Problem:
        """Load the PEtab problem."""
        if self._petab_problem is not None:
            return self._petab_problem
        with tempfile.TemporaryDirectory() as temp_dir:
            download_success = self.download_files(temp_dir)
            if not download_success:
                raise FileNotFoundError(
                    f"Could not download files from {self.github_repo}. "
                    f"Check your internet connection."
                )
            self._petab_problem = petab.Problem.from_yaml(
                os.path.join(temp_dir, self.petab_yaml)
            )
        return self._petab_problem

    @property
    def importer(self) -> PetabImporter:
        """Load the importer."""
        if self._importer is None:
            self._importer = PetabImporter(
                self.petab_problem, hierarchical=self.hierarchical
            )
        return self._importer

    @property
    def objective(self) -> ObjectiveBase:
        """Load the objective function."""
        if self._objective is None:
            self._objective = self.problem.objective
        return self._objective

    @property
    def problem(self) -> Problem:
        """Load the problem."""
        if self._problem is None:
            self._problem = self.importer.create_problem()
        return self._problem

    def download_files(self, dir: str):
        """Download the petab files from the github repository to ``dir``."""
        for filename in self.filenames:
            url = f"{self.github_repo}/{filename}"
            response = requests.get(url, timeout=10)
            if response.status_code != 200:
                return False
            with open(os.path.join(dir, filename), "wb") as file:
                file.write(response.content)
        return True
