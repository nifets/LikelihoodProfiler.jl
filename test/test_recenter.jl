using Test, LikelihoodProfiler, OptimizationLBFGSB, ForwardDiff

tilted_well(x, p) = (x[2]^2 - 1)^2 + 0.2 * x[2] + (x[1] - x[2])^2

optf = OptimizationFunction(tilted_well, AutoForwardDiff())
worse_optimum = solve(OptimizationProblem(optf, [1.0, 1.0]), LBFGSB()).u
better_optimum = solve(OptimizationProblem(optf, [-1.0, -1.0]), LBFGSB()).u

plprob = ProfileLikelihoodProblem(OptimizationProblem(optf, worse_optimum), worse_optimum;
  profile_lower = -3.0, profile_upper = 3.0, threshold = 2.0)
method = OptimizationProfiler(; optimizer = LBFGSB(), stepper = FixedStep(; initial_step = 0.05))

@testset "Default does not recenter" begin
  sol = solve(plprob, method)
  @test sol.prob.optpars == worse_optimum
  @test all(obj_level.(sol.profiles) .≈ tilted_well(worse_optimum, nothing) + 2.0)
  @test minimum(minimum(profile.obj) for profile in sol.profiles) < tilted_well(worse_optimum, nothing)
end

@testset "Recenter on better optimum" begin
  sol = @test_logs (:warn, r"below the optimum") match_mode=:any solve(plprob, method; max_recenters = 1)
  @test sol.prob.optpars ≈ better_optimum atol = 1e-4
  @test all(obj_level.(sol.profiles) .≈ tilted_well(better_optimum, nothing) + 2.0)
  @test minimum(minimum(profile.obj) for profile in sol.profiles) >= tilted_well(better_optimum, nothing) - 1e-3
end

@testset "Recentering requires an optimizer" begin
  @test_throws ArgumentError solve(plprob, QuadraticApproxProfiler(); max_recenters = 1)
end
