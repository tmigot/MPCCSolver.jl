"""
Package de fonctions de pénalité

liste des fonctions :

fonctions additionnelles :
_rho_detail(rho::Vector,mp::AbstractNLPModel,ncc::Int64)
_err_detail(err::Vector,n::Int64,ncc::Int64,ncon::Int64)
"""
module Penalty

import NLPModels: AbstractNLPModel, hess, jtprod, jac

###################################
#
#Fonction de pénalité quadratique
#
###################################

include("penalty_quadratic.jl")

###################################
#
#Fonction de pénalité Lagrangienne
#
###################################

include("penalty_lagrangian.jl")

"""
Renvoie : rho_eq, rho_ineq_var, rho_ineq_cons
"""
function _rho_detail(rho :: Vector,
                     mp  :: AbstractNLPModel,
                     ncc :: Int64)

 nil  = length(mp.meta.lvar)
 niu  = length(mp.meta.uvar)
 nlcon = length(mp.meta.lcon)
 nucon = length(mp.meta.ucon)

 return rho[1:ncc],rho[ncc+1:2*ncc],rho[2*ncc+1:2*ncc+nil],
        rho[2*ncc+nil+1:2*ncc+nil+niu],
        rho[2*ncc+nil+niu+1:2*ncc+nil+niu+nlcon],
        rho[2*ncc+nil+niu+nlcon+1:2*ncc+nil+niu+nlcon+nucon]
end

function _err_detail(err::Vector,n::Int64,ncc::Int64,ncon::Int64)

 if ncc>0
  err_eq_g=err[1:ncc]
  err_eq_h=err[ncc+1:2*ncc]
 else
  err_eq_g=Array{Float64}(0)
  err_eq_h=Array{Float64}(0)
 end

 err_in_lv=err[2*ncc+1:2*ncc+n]
 err_in_uv=err[2*ncc+n+1:2*ncc+2*n]

 if ncon>0
  err_in_lc=err[2*ncc+2*n+1:2*ncc+2*n+ncon]
  err_in_uc=err[2*ncc+2*n+ncon+1:2*ncc+2*n+2*ncon]
 else
  err_in_lc=Array{Float64}(0)
  err_in_uc=Array{Float64}(0)
 end

 return err_eq_g,err_eq_h,err_in_lv,err_in_uv,err_in_lc,err_in_uc
end

#end of module
end
