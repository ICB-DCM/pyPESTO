module MyPEtabJlModule

using OrdinaryDiffEq
using Sundials
using PEtab

pathYaml = "/Users/pauljonasjost/Documents/GitHub_Folders/pyPESTO/test/julia/../../doc/example/conversion_reaction/conversion_reaction.yaml"
petabModel = PEtabModel(pathYaml, verbose=true)

# A full list of options for PEtabODEProblem can be found at https://sebapersson.github.io/PEtab.jl/stable/API_choosen/
petabProblem = PEtabODEProblem(
	petabModel,
	ode_solver=ODESolver(Rodas5P(), abstol=1e-08, reltol=1e-08, maxiters=Int64(1e4)),
	gradient_method=:ForwardDiff,
	hessian_method=:ForwardDiff,
	sparse_jacobian=nothing,
	verbose=true
)

end
