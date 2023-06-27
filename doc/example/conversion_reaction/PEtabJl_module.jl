module MyPEtabJlModule

using OrdinaryDiffEq
using PEtab

pathYaml = "doc/example/conversion_reaction/conversion_reaction.yaml"
petabModel = readPEtabModel(pathYaml, verbose=true)

# A full list of options for createPEtabODEProblem can be found at https://sebapersson.github.io/PEtab.jl/dev/API_choosen/#PEtab.setupPEtabODEProblem
petabProblem = createPEtabODEProblem(
	petabModel,
	odeSolverOptions=ODESolverOptions(Rodas5P(), abstol=1e-08, reltol=1e-08, maxiters=Int64(1e4)),
	gradientMethod=:ForwardDiff,
	hessianMethod=:ForwardDiff,
	sparseJacobian=false,
	verbose=true
)

end
