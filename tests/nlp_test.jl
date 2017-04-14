gc()

#Tests des packages
include("../ALAS/include.jl")

using MPCCmod
using OutputRelaxationmod

using CUTEst

nlp = CUTEstModel("ZY2");
print(nlp);

#déclare le mpcc :
exemple_nlp=MPCCmod.MPCC(nlp)

println("Butterfly method:")
#résolution avec ALAS Butterfly
xb,fb,orb = MPCCsolve.solve(exemple_nlp)

finalize(nlp)
