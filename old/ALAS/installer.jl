if Pkg.installed("JuMP") == nothing
    Pkg.add("JuMP")
end

if Pkg.installed("NLPModels") == nothing
    Pkg.add("NLPModels")
end

if Pkg.installed("Ipopt") == nothing
    Pkg.add("Ipopt") #nécessite un compilateur fortran (gfortran)
end

#if Pkg.installed("PyPlot") == nothing
#    Pkg.add("PyPlot") #nécessite Python d'installer
#end

if Pkg.installed("BenchmarkProfiles") == nothing
    Pkg.add("BenchmarkProfiles")
end

if Pkg.installed("CUTEst") == nothing
    Pkg.add("CUTEst") #nécessite gfortran, wget, gsl
end

if Pkg.installed("OptimizationProblems") == nothing
 #using OptimizationProblems
 Pkg.clone("https://github.com/JuliaSmoothOptimizers/OptimizationProblems.jl.git")
end

if Pkg.installed("AmplNLReader") == nothing
 Pkg.clone("https://github.com/JuliaSmoothOptimizers/AmplNLReader.jl.git")
 Pkg.build("AmplNLReader") #nécessite cmake
end
