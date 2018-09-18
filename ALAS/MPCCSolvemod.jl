"""
MPCCSolve(mod::MPCCmod.MPCC)
"""

module MPCCSolvemod

import MPCCmod: MPCC
import ParamSetmod.ParamSet
import ParamMPCCmod.ParamMPCC
import AlgoSetmod.AlgoSet
import RMPCCmod: RMPCC, start!, update!, final_update!
import StoppingMPCCmod: StoppingMPCC, stop!, stop_start!, final
import OutputRelaxationmod: OutputRelaxation, UpdateOR, Print, final!
import RlxMPCCSolvemod: RlxMPCCSolve, update_rlx!
import RRelaxmod.relax_start!

type MPCCSolve

 mod        :: MPCC

 name_relax :: AbstractString

 xj         :: Vector #itéré courant

 algoset    :: AlgoSet

 paramset   :: ParamSet
 parammpcc  :: ParamMPCC

 rmpcc      :: RMPCC
 smpcc      :: StoppingMPCC

end

function MPCCSolve(mod        :: MPCC,
                   x          :: Vector;
                   name_relax :: String    = "KS",
                   algoset    :: AlgoSet   = AlgoSet(),
                   paramset   :: ParamSet  = ParamSet(2*(mod.meta.nvar + mod.meta.ncon + mod.meta.ncc)),
                   parammpcc  :: ParamMPCC = ParamMPCC(2*(mod.meta.nvar + mod.meta.ncon + mod.meta.ncc)))

 rmpcc = RMPCC(x)
 smpcc = StoppingMPCC(precmpcc    = parammpcc.precmpcc,
                      paramin     = parammpcc.paramin,
                      prec_oracle = parammpcc.prec_oracle)

 return MPCCSolve(mod, name_relax, x, algoset, paramset, parammpcc, rmpcc, smpcc)
end

"""
Accesseur : modifie le point initial
"""

function set_x(mod :: MPCCSolve,
               x0  :: Vector)

 mod.xj = x0

 return mod
end

###################################################################################
#
# MAIN FUNCTION: solve
#
###################################################################################

include("mpcc_solve.jl")

"""
Une méthode supplémentaire rapide
"""
function solve(mod :: MPCC)

 return solve(MPCCSolve(mod, mod.mp.meta.x0))
end



#end of module
end
