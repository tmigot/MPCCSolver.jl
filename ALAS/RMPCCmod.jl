module RMPCCmod

import MPCCmod.MPCC, MPCCmod.obj, MPCCmod.grad
import MPCCmod.jac_actif
import MPCCmod.viol_contrainte, MPCCmod.viol_comp, MPCCmod.viol_cons

import NLPModels.cons

type RMPCC

 x :: Vector
 fx :: Float64 #objective at x
 gx :: Vector #gradient at x

 feas :: Vector #violation of the constraints
 feas_cc :: Vector #violation of the constraints
 norm_feas :: Float64 # norm of feas

 dual_feas :: Vector
 norm_dual_feas :: Float64

 lambda :: Vector

 solved :: Int64 #code de sortie

end

function RMPCC(x              :: Vector;
               fx             :: Float64 = Inf,
               gx             :: Vector  = Float64[],
               feas           :: Vector  = Float64[],
               feas_cc        :: Vector  = Float64[],
               norm_feas      :: Float64 = Inf,
               dual_feas      :: Vector  = Float64[],
               norm_dual_feas :: Float64 = Inf,
               lambda         :: Vector  = Float64[],
               solved         :: Int64   = -1)


 return RMPCC(x,fx,gx,feas,feas_cc,norm_feas,dual_feas,norm_dual_feas,lambda,solved)
end

function start!(rmpcc :: RMPCC,
               mod :: MPCC,
               xk :: Vector)

 rmpcc.fx = obj(mod,xk)

 rmpcc.feas = viol_cons(mod,xk)
 rmpcc.feas_cc = viol_comp(mod,xk)
 rmpcc.norm_feas = norm(vcat(rmpcc.feas,rmpcc.feas_cc),Inf)

 return rmpcc
end

function update!(rmpcc          :: RMPCC,
                 mod            :: MPCC,
                 xk             :: Vector;
                 fx             :: Float64 = Inf,
                 gx             :: Vector  = Float64[],
                 feas           :: Vector  = Float64[],
                 norm_feas      :: Float64 = Inf,
                 dual_feas      :: Vector  = Float64[],
                 norm_dual_feas :: Float64 = Inf,
                 lambda         :: Vector  = Float64[])

 rmpcc.fx = fx == Inf ? obj(mod,xk) : fx
 rmpcc.gx = gx == Float64[] ? rmpcc.gx : gx

 rmpcc.feas = feas == Float64[] ? rmpcc.feas : feas
 rmpcc.norm_feas = norm_feas == Inf ? viol_contrainte_norm(mod,xk,tnorm=Inf) : norm_feas
 rmpcc.dual_feas = dual_feas == Float64[] ? rmpcc.dual_feas : dual_feas
 rmpcc.norm_dual_feas = norm_dual_feas == Inf ? rmpcc.norm_dual_feas : norm_feas

 rmpcc.lambda = lambda == Float64[] ? rmpcc.lambda : lambda

 return rmpcc
end

##################################################################################
#
# Additional functions
#
##################################################################################

"""
Donne la norme 2 de la violation des contraintes avec slack

note : devrait appeler viol_contrainte
"""
function viol_contrainte_norm(mod   :: MPCC,
                              x     :: Vector,
                              yg    :: Vector,
                              yh    :: Vector;
                              tnorm :: Real=2)

 return norm(viol_contrainte(mod,x,yg,yh),tnorm)
end

#x de taille n+2nb_comp
function viol_contrainte_norm(mod   :: MPCC,
                              x     :: Vector;
                              tnorm :: Real=2)

 n=mod.n

 if length(x)==n
  resul=max(viol_comp_norm(mod,x),viol_cons_norm(mod,x))
 else
  resul=viol_contrainte_norm(mod,x[1:n],x[n+1:n+mod.nb_comp],x[n+mod.nb_comp+1:n+2*mod.nb_comp],tnorm=tnorm)
 end

 return resul
end

"""
Donne la norme de la violation de la complémentarité min(G,H)
"""
function viol_comp_norm(mod   :: MPCC,
                        x     :: Vector;
                        tnorm :: Real=2)

 return mod.nb_comp > 0 ? norm(viol_comp(mod,x),tnorm) : 0
end

"""
Donne la norme de la violation des contraintes \"classiques\"
"""
function viol_cons_norm(mod   :: MPCC,
                        x     :: Vector;
                        tnorm :: Real=2)

 n=mod.n
 return norm(viol_cons(mod,x),Inf)
end

#function viol_cons_norm(mod   :: MPCC,
#                        x     :: Vector;
#                        tnorm :: Real=2)

# n=mod.n

# x=length(x)==n?x:x[1:n]

# feas=norm([max.(mod.mp.meta.lvar-x,0);max.(x-mod.mp.meta.uvar,0)],tnorm)

# if mod.mp.meta.ncon !=0

#  c=cons(mod.mp,x)
#  feas=max(norm([max.(mod.mp.meta.lcon-c,0);max.(c-mod.mp.meta.ucon,0)],tnorm),feas)

# end

# return feas
#end


#end of module
end
