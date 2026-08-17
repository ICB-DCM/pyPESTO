from .example_class import PyPESTOExamplePEtab

boehm = PyPESTOExamplePEtab(
    name="boehm_JProteomeRes2014",
    description="A model of the STAT5 dimerization. Based on the publication "
    "by Boehm et al. (2014) https://doi.org/10.1021/pr5006923",
    github_repo="https://raw.githubusercontent.com/ICB-DCM/pyPESTO/refs/heads/"
    "main/doc/example/boehm_JProteomeRes2014",
    filenames=[
        "boehm_JProteomeRes2014.yaml",
        "parameters_Boehm_JProteomeRes2014.tsv",
        "experimentalCondition_Boehm_JProteomeRes2014.tsv",
        "measurementData_Boehm_JProteomeRes2014.tsv",
        "observables_Boehm_JProteomeRes2014.tsv",
        "boehm_JProteomeRes2014.xml",
        "visualizationSpecification_Boehm_JProteomeRes2014.tsv",
    ],
)

censored_data = PyPESTOExamplePEtab(
    name="censored_data",
    description="A model with censored data. Right censoring >20, left <3, "
    "interval censoring [10, 16].",
    github_repo="https://raw.githubusercontent.com/ICB-DCM/pyPESTO/refs/heads/"
    "main/doc/example/example_censored",
    filenames=[
        "example_censored.yaml",
        "parameters_example_censored.tsv",
        "experimentalCondition_example_censored.tsv",
        "measurementData_example_censored.tsv",
        "observables_example_censored.tsv",
        "model_example_censored.xml",
    ],
    detailed_description="This model was taken from "
    "https://doi.org/10.1093/bioinformatics/btab512 and "
    "censoring was added to it. The PEtab problem was "
    "adjusted accordingly.",
    hierarchical=True,
)

ordinal_data = PyPESTOExamplePEtab(
    name="ordinal_data",
    description="A model with ordinal data. The data is binned into three "
    "categories.",
    github_repo="https://raw.githubusercontent.com/ICB-DCM/pyPESTO/refs/heads/"
    "main/doc/example/example_ordinal",
    filenames=[
        "example_ordinal.yaml",
        "parameters_example_ordinal.tsv",
        "experimentalCondition_example_ordinal.tsv",
        "measurementData_example_ordinal.tsv",
        "observables_example_ordinal.tsv",
        "model_example_ordinal.xml",
    ],
    detailed_description="This model was taken from "
    "https://doi.org/10.1093/bioinformatics/btab512 and "
    "The PEtab problem was slightly adjusted.",
    hierarchical=True,
)

semiquantitative_data = PyPESTOExamplePEtab(
    name="semiquantitative_data",
    description="A model with semiquantitative data. Nonlinear "
    "transformations were applied to the data.",
    github_repo="https://raw.githubusercontent.com/ICB-DCM/pyPESTO/refs/heads/"
    "main/doc/example/example_semiquantitative",
    filenames=[
        "example_semiquantitative.yaml",
        "parameters_example_semiquantitative.tsv",
        "experimentalCondition_example_semiquantitative.tsv",
        "measurementData_example_semiquantitative.tsv",
        "observables_example_semiquantitative.tsv",
        "model_example_semiquantitative.xml",
    ],
    detailed_description="This model was taken from "
    "https://doi.org/10.1093/bioinformatics/btab512 and "
    "the PEtab problem was adjusted.",
    hierarchical=True,
)

semiquantitative_data_linear = PyPESTOExamplePEtab(
    name="semiquantitative_data_linear",
    description="A model with semiquantitative data. Linear "
    "transformations were applied to the data.",
    github_repo="https://raw.githubusercontent.com/ICB-DCM/pyPESTO/refs/heads/"
    "main/doc/example/example_semiquantitative",
    filenames=[
        "example_semiquantitative_linear.yaml",
        "parameters_example_semiquantitative.tsv",
        "experimentalCondition_example_semiquantitative.tsv",
        "measurementData_example_semiquantitative_linear.tsv",
        "observables_example_semiquantitative.tsv",
        "model_example_semiquantitative.xml",
    ],
    detailed_description="This model was taken from "
    "https://doi.org/10.1093/bioinformatics/btab512 and "
    "the PEtab problem was adjusted.",
    hierarchical=True,
)

conversion_reaction = PyPESTOExamplePEtab(
    name="conversion_reaction",
    description="A simple conversion reaction model. A -> B and B -> A.",
    github_repo="https://raw.githubusercontent.com/ICB-DCM/pyPESTO/refs/heads/"
    "main/doc/example/conversion_reaction",
    filenames=[
        "conversion_reaction.yaml",
        "parameters.tsv",
        "conditions.tsv",
        "measurements.tsv",
        "observables.tsv",
        "model_conversion_reaction.xml",
    ],
)
