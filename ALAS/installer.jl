if !Plots.is_installed("JuMP")
    Pkg.add("JuMP")
end

if !Plots.is_installed("NLPModels")
    Pkg.add("NLPModels")
end

if !Plots.is_installed("Ipopt")
    Pkg.add("Ipopt") #nécessite un compilateur fortran (gfortran)
end

#if !Plots.is_installed("PyPlot")
#    Pkg.add("PyPlot") #nécessite Python d'installer
#end

if !Plots.is_installed("BenchmarkProfiles")
    Pkg.add("BenchmarkProfiles")
end

if !Plots.is_installed("CUTEst")
    Pkg.add("CUTEst") #nécessite gfortran, wget, gsl
end

#using OptimizationProblems
Pkg.clone("https://github.com/JuliaSmoothOptimizers/OptimizationProblems.jl.git")


