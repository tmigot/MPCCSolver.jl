"""
Package de fonctions de pénalité

liste des fonctions :
Quadratic(mp::AbstractNLPModel,err::Vector,ncc::Int64,x::Vector,yg::Vector,
                   yh::Vector,rho::Vector,usg::Vector,ush::Vector,uxl::Vector,uxu::Vector,ucl::Vector,ucu::Vector)
Quadratic(mp::AbstractNLPModel,err::Vector,ncc::Int64,x::Vector,yg::Vector,
                   yh::Vector,rho::Vector,usg::Vector,ush::Vector,uxl::Vector,uxu::Vector,ucl::Vector,ucu::Vector,f::Float64,rho_update::Float64)
Quadratic(mp::AbstractNLPModel,G::AbstractNLPModel,
                   H::AbstractNLPModel,err::Vector,ncc::Int64,
                   x::Vector,yg::Vector,yh::Vector,
                   rho::Vector,
                   usg::Vector,ush::Vector,
                   uxl::Vector,uxu::Vector,
                   ucl::Vector,ucu::Vector,
                   g::Vector)
Quadratic(mp::AbstractNLPModel,G::AbstractNLPModel,
                   H::AbstractNLPModel,err::Vector,ncc::Int64,
                   x::Vector,yg::Vector,yh::Vector,
                   rho::Vector,
                   usg::Vector,ush::Vector,
                   uxl::Vector,uxu::Vector,
                   ucl::Vector,ucu::Vector,
                   g::Vector,rho_update::Float64)
Quadratic(mp::AbstractNLPModel,G::AbstractNLPModel,
                   H::AbstractNLPModel,err::Vector,ncc::Int64,
                   x::Vector,yg::Vector,yh::Vector,
                   rho::Vector,
                   usg::Vector,ush::Vector,
                   uxl::Vector,uxu::Vector,
                   ucl::Vector,ucu::Vector,
                   Hess::Array{Float64,2})
Quadratic(mp::AbstractNLPModel,G::AbstractNLPModel,
                   H::AbstractNLPModel,err::Vector,ncc::Int64,
                   x::Vector,yg::Vector,yh::Vector,
                   rho::Vector,
                   usg::Vector,ush::Vector,
                   uxl::Vector,uxu::Vector,
                   ucl::Vector,ucu::Vector,
                   Hess::Array{Float64,2},rho_update::Float64)


Lagrangian(mp::AbstractNLPModel,err::Vector,ncc::Int64,x::Vector,yg::Vector,
                   yh::Vector,rho::Vector,usg::Vector,ush::Vector,uxl::Vector,uxu::Vector,ucl::Vector,ucu::Vector)
Lagrangian(mp::AbstractNLPModel,G::AbstractNLPModel,
                   H::AbstractNLPModel,err::Vector,ncc::Int64,
                   x::Vector,yg::Vector,yh::Vector,
                   rho::Vector,
                   usg::Vector,ush::Vector,
                   uxl::Vector,uxu::Vector,
                   ucl::Vector,ucu::Vector,
                   g::Vector)
Lagrangian(mp::AbstractNLPModel,G::AbstractNLPModel,
                   H::AbstractNLPModel,err::Vector,ncc::Int64,
                   x::Vector,yg::Vector,yh::Vector,
                   rho::Vector,
                   usg::Vector,ush::Vector,
                   uxl::Vector,uxu::Vector,
                   ucl::Vector,ucu::Vector,
                   Hess::Array{Float64,2})

fonctions additionnelles :
_rho_detail(rho::Vector,mp::AbstractNLPModel,ncc::Int64)
_err_detail(err::Vector,n::Int64,ncc::Int64,ncon::Int64)
"""
module Penalty

import NLPModels.AbstractNLPModel
import NLPModels.hess
import NLPModels.jtprod
import NLPModels.jac

###################################
#
#Fonction de pénalité quadratique
#
###################################

include("penalty_quadratic.jl")

###################################
#
#Fonction de pénalité quadratique
#
###################################
"""
Fonction de pénalité lagrangienne
"""
function lagrangian(mp::AbstractNLPModel,err::Vector,ncc::Int64,x::Vector,yg::Vector,
                   yh::Vector,rho::Vector,usg::Vector,ush::Vector,uxl::Vector,uxu::Vector,ucl::Vector,ucu::Vector)
 n=length(mp.meta.x0)
 #mod::MPCCmod.MPCC,r::Float64,s::Float64,t::Float64
 rho_eqg,rho_eqh,rho_ineq_lvar,rho_ineq_uvar,rho_ineq_lcons,rho_ineq_ucons=_rho_detail(rho,mp,ncc)
 err_eq_g,err_eq_h,err_in_lv,err_in_uv,err_in_lc,err_in_uc=_err_detail(err,n,ncc,mp.meta.ncon)

 lagrangian=dot(err_eq_g,usg)+dot(err_eq_h,ush)
# Pen_eq2=rhof*norm((G(x)-yg))^2+rhof*norm((H(x)-yh))^2
 Pen_eq=dot(rho_eqg.*err_eq_g,err_eq_g)+dot(rho_eqh.*err_eq_h,err_eq_h)
# Pen_in_lv=norm(sqrt(rho_ineq_lvar).*max(mp.meta.lvar-x+uxl./rho_ineq_lvar,0.0))^2
 Pen_in_lv=dot(rho_ineq_lvar.*err_in_lv,err_in_lv)
# Pen_in_uv=norm(sqrt(rho_ineq_uvar).*max(x-mp.meta.uvar+uxu./rho_ineq_uvar,0.0))^2
 Pen_in_uv=dot(rho_ineq_uvar.*err_in_uv,err_in_uv)
# Pen_in_lc=norm(sqrt(rho_ineq_lcons).*max(mp.meta.lcon-mp.c(x)+ucl./rho_ineq_lcons,0.0))^2
 Pen_in_lc=dot(rho_ineq_lcons.*err_in_lc,err_in_lc)
 Pen_in_uc=dot(rho_ineq_ucons.*err_in_uc,err_in_uc)

 return lagrangian+0.5*(Pen_eq+Pen_in_lv+Pen_in_uv+Pen_in_lc+Pen_in_uc)
end

function lagrangian(mp::AbstractNLPModel,G::AbstractNLPModel,
                   H::AbstractNLPModel,err::Vector,ncc::Int64,
                   x::Vector,yg::Vector,yh::Vector,
                   rho::Vector,
                   usg::Vector,ush::Vector,
                   uxl::Vector,uxu::Vector,
                   ucl::Vector,ucu::Vector,
                   g::Vector)
 n=length(mp.meta.x0)

 rho_eqg,rho_eqh,rho_ineq_lvar,rho_ineq_uvar,rho_ineq_lcons,rho_ineq_ucons=_rho_detail(rho,mp,ncc)
 err_eq_g,err_eq_h,err_in_lv,err_in_uv,err_in_lc,err_in_uc=_err_detail(err,n,ncc,mp.meta.ncon)

  G_in_lv=-rho_ineq_lvar.*err_in_lv
  G_in_uv=rho_ineq_uvar.*err_in_uv

  if mp.meta.ncon!=0
   Jtv(z)=jtprod(mp,x,z)
   G_in_lc=-Jtv(rho_ineq_lcons.*err_in_lc)
   G_in_uc=Jtv(rho_ineq_ucons.*err_in_uc)
  else
   G_in_uc=zeros(n)
   G_in_lc=zeros(n)
  end

  if ncc>0
   G_eq=jtprod(G,x,rho_eqg.*err_eq_g+usg)+jtprod(H,x,rho_eqh.*err_eq_h+ush)
   return vec([(G_eq+G_in_lv+G_in_uv+G_in_lc+G_in_uc)' (-rho_eqg.*err_eq_g)' (-rho_eqh.*err_eq_h)'])
  else
   return G_in_lv+G_in_uv+G_in_lc+G_in_uc
  end
end

function lagrangian(mp::AbstractNLPModel,G::AbstractNLPModel,
                   H::AbstractNLPModel,err::Vector,ncc::Int64,
                   x::Vector,yg::Vector,yh::Vector,
                   rho::Vector,
                   usg::Vector,ush::Vector,
                   uxl::Vector,uxu::Vector,
                   ucl::Vector,ucu::Vector,
                   Hess::Array{Float64,2})

 n=length(mp.meta.x0)

 rho_eqg,rho_eqh,rho_ineq_lvar,rho_ineq_uvar,rho_ineq_lcons,rho_ineq_ucons=_rho_detail(rho,mp,ncc)
 err_eq_g,err_eq_h,err_in_lv,err_in_uv,err_in_lc,err_in_uc=_err_detail(err,n,ncc,mp.meta.ncon)

  lv=zeros(n);ilv=find(x->x>0,err_in_lv);lv[ilv]=ones(length(ilv))
  uv=zeros(n);iuv=find(x->x>0,err_in_uv);uv[iuv]=ones(length(iuv))

  if ncc>0 && mp.meta.ncon!=0
   ilc=find(x->x>0,err_in_lc);rho_ineq_lcons[ilc]=zeros(length(ilc))
   iuc=find(x->x>0,err_in_uc);rho_ineq_ucons[iuc]=zeros(length(iuc))
   Hlc=hess(mp, x, obj_weight=1.0, y=rho_ineq_lcons.*err_in_lc)+(diagm(rho_ineq_lcons)*jac(mp,x))'*jac(mp,x)
   Huc=hess(mp, x, obj_weight=1.0, y=rho_ineq_ucons.*err_in_uc)+(diagm(rho_ineq_ucons)*jac(mp,x))'*jac(mp,x)

   HCG=hess(G, x, obj_weight=1.0, y=err_eq_g+usg)+(diagm(rho_eqg)*jac(G,x))'*jac(G,x)
   HCH=hess(H, x, obj_weight=1.0, y=err_eq_h+ush)+(diagm(rho_eqh)*jac(H,x))'*jac(H,x)

   return tril([diagm(rho_ineq_lvar.*lv)+diagm(rho_ineq_uvar.*uv)+HCG+HCH+Hlc+Huc zeros(n,2*ncc);zeros(2*ncc,n) diagm([rho_eqg;rho_eqh])])
  elseif ncc>0
   HCG=hess(G, x, obj_weight=1.0, y=err_eq_g+usg)+(diagm(rho_eqg)*jac(G,x))'*jac(G,x)
   HCH=hess(H, x, obj_weight=1.0, y=err_eq_h+ush)+(diagm(rho_eqh)*jac(H,x))'*jac(H,x)
   return tril([diagm(rho_ineq_lvar.*lv)+diagm(rho_ineq_uvar.*uv)+HCG+HCH zeros(n,n);zeros(2*ncc,n) diagm([rho_eqg;rho_eqh])])
  elseif mp.meta.ncon!=0
   ilc=find(x->x>0,err_in_lc);rho_ineq_lcons[ilc]=zeros(length(ilc))
   iuc=find(x->x>0,err_in_uc);rho_ineq_ucons[iuc]=zeros(length(iuc))

   Hlc=hess(mp, x, obj_weight=1.0, y=rho_ineq_lcons.*err_in_lc)+(diagm(rho_ineq_lcons)*jac(mp,x))'*jac(mp,x)
   Huc=hess(mp, x, obj_weight=1.0, y=rho_ineq_ucons.*err_in_uc)+(diagm(rho_ineq_ucons)*jac(mp,x))'*jac(mp,x)
   return tril(diagm(rho_ineq_lvar.*lv)+diagm(rho_ineq_uvar.*uv)+Hlc+Huc)
  else
   return tril(diagm(rho_ineq_lvar.*lv)+diagm(rho_ineq_uvar.*uv))
  end
end

"""
Renvoie : rho_eq, rho_ineq_var, rho_ineq_cons
"""
function _rho_detail(rho::Vector,mp::AbstractNLPModel,ncc::Int64)
 nb_ineq_lvar=length(mp.meta.lvar)
 nb_ineq_uvar=length(mp.meta.uvar)
 nb_ineq_lcons=length(mp.meta.lcon)
 nb_ineq_ucons=length(mp.meta.ucon)

 return rho[1:ncc],rho[ncc+1:2*ncc],rho[2*ncc+1:2*ncc+nb_ineq_lvar],
        rho[2*ncc+nb_ineq_lvar+1:2*ncc+nb_ineq_lvar+nb_ineq_uvar],
        rho[2*ncc+nb_ineq_lvar+nb_ineq_uvar+1:2*ncc+nb_ineq_lvar+nb_ineq_uvar+nb_ineq_lcons],
        rho[2*ncc+nb_ineq_lvar+nb_ineq_uvar+nb_ineq_lcons+1:2*ncc+nb_ineq_lvar+nb_ineq_uvar+nb_ineq_lcons+nb_ineq_ucons]
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
