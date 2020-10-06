#Benchmark profile

using BenchmarkProfiles
T=10*rand(25,3) #3 solvers, 25 problemes
performance_profile(T,["Solver 1","Sovler 2","Sovler 3"], title="Celebrity Death Match")
