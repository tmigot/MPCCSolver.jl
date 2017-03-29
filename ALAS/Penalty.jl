"""
Package de fonctions de pénalité

liste des fonctions :
function Quadratic(mp::NLPModels.AbstractNLPModel,G::Function,H::Function,nb_comp::Int64,x::Vector,yg::Vector,yh::Vector,rho::Vector,usg::Vector,ush::Vector,uxl::Vector,uxu::Vector,ucl::Vector,ucu::Vector)
Lagrangian(mp::NLPModels.AbstractNLPModel,G::Function,H::Function,nb_comp::Int64,x::Vector,yg::Vector,yh::Vector,rho::Vector,usg::Vector,ush::Vector,uxl::Vector,uxu::Vector,ucl::Vector,ucu::Vector)

fonctions additionnelles :
RhoDetail(rho::Vector,mp::NLPModels.AbstractNLPModel,nb_comp::Int64)
"""
module Penalty

using NLPModels

"""
Fonction de pénalité quadratique
"""
function Quadratic(mp::NLPModels.AbstractNLPModel,G::Function,H::Function,nb_comp::Int64,x::Vector,yg::Vector,yh::Vector,rho::Vector,usg::Vector,ush::Vector,uxl::Vector,uxu::Vector,ucl::Vector,ucu::Vector)
 #mod::MPCCmod.MPCC,r::Float64,s::Float64,t::Float64
 rho_eqg,rho_eqh,rho_ineq_lvar,rho_ineq_uvar,rho_ineq_lcons,rho_ineq_ucons=RhoDetail(rho,mp,nb_comp)

# Pen_eq2=rhof*norm((alas.mod.G(x)-yg))^2+rhof*norm((alas.mod.H(x)-yh))^2
 err_eq_g=(G(x)-yg)
 err_eq_h=(H(x)-yh)
 Pen_eq=dot(rho_eqg.*err_eq_g,err_eq_g)+dot(rho_eqh.*err_eq_h,err_eq_h)
# Pen_in_lv=norm(sqrt(rho_ineq_lvar).*max(mod.mp.meta.lvar-x+uxl./rho_ineq_lvar,0.0))^2
 err_in_lv=max(mp.meta.lvar-x+uxl./rho_ineq_lvar,0.0)
 Pen_in_lv=dot(rho_ineq_lvar.*err_in_lv,err_in_lv)
# Pen_in_uv=norm(sqrt(rho_ineq_uvar).*max(x-mod.mp.meta.uvar+uxu./rho_ineq_uvar,0.0))^2
 err_in_uv=max(x-mp.meta.uvar+uxu./rho_ineq_uvar,0.0)
 Pen_in_uv=dot(rho_ineq_uvar.*err_in_uv,err_in_uv)
# Pen_in_lc=norm(sqrt(rho_ineq_lcons).*max(mod.mp.meta.lcon-alas.mod.mp.c(x)+ucl./rho_ineq_lcons,0.0))^2
 err_in_lc=max(mp.meta.lcon-mp.c(x)+ucl./rho_ineq_lcons,0.0)
 Pen_in_lc=dot(rho_ineq_lcons.*err_in_lc,err_in_lc)
 err_in_uc=max(mp.c(x)-mp.meta.ucon+ucu./rho_ineq_ucons,0.0)
 Pen_in_uc=dot(rho_ineq_ucons.*err_in_uc,err_in_uc)

 return mp.f(x)+0.5*(Pen_eq+Pen_in_lv+Pen_in_uv+Pen_in_lc+Pen_in_uc)
end

"""
Fonction de pénalité lagrangienne
"""
function Lagrangian(mp::NLPModels.AbstractNLPModel,G::Function,H::Function,nb_comp::Int64,x::Vector,yg::Vector,yh::Vector,rho::Vector,usg::Vector,ush::Vector,uxl::Vector,uxu::Vector,ucl::Vector,ucu::Vector)
 #mod::MPCCmod.MPCC,r::Float64,s::Float64,t::Float64
 rho_eqg,rho_eqh,rho_ineq_lvar,rho_ineq_uvar,rho_ineq_lcons,rho_ineq_ucons=RhoDetail(rho,mp,nb_comp)

 Lagrangian=dot(mod.G(x)-yg,usg)+dot(mod.H(x)-yh,ush)
# Pen_eq2=rhof*norm((alas.mod.G(x)-yg))^2+rhof*norm((alas.mod.H(x)-yh))^2
 err_eq_g=(G(x)-yg)
 err_eq_h=(H(x)-yh)
 Pen_eq=dot(rho_eqg.*err_eq_g,err_eq_g)+dot(rho_eqh.*err_eq_h,err_eq_h)
# Pen_in_lv=norm(sqrt(rho_ineq_lvar).*max(mod.mp.meta.lvar-x+uxl./rho_ineq_lvar,0.0))^2
 err_in_lv=max(mp.meta.lvar-x+uxl./rho_ineq_lvar,0.0)
 Pen_in_lv=dot(rho_ineq_lvar.*err_in_lv,err_in_lv)
# Pen_in_uv=norm(sqrt(rho_ineq_uvar).*max(x-mod.mp.meta.uvar+uxu./rho_ineq_uvar,0.0))^2
 err_in_uv=max(x-mp.meta.uvar+uxu./rho_ineq_uvar,0.0)
 Pen_in_uv=dot(rho_ineq_uvar.*err_in_uv,err_in_uv)
# Pen_in_lc=norm(sqrt(rho_ineq_lcons).*max(mod.mp.meta.lcon-alas.mod.mp.c(x)+ucl./rho_ineq_lcons,0.0))^2
 err_in_lc=max(mp.meta.lcon-mp.c(x)+ucl./rho_ineq_lcons,0.0)
 Pen_in_lc=dot(rho_ineq_lcons.*err_in_lc,err_in_lc)
 err_in_uc=max(mp.c(x)-mp.meta.ucon+ucu./rho_ineq_ucons,0.0)
 Pen_in_uc=dot(rho_ineq_ucons.*err_in_uc,err_in_uc)

 return mp.f(x)+Lagrangian+0.5*(Pen_eq+Pen_in_lv+Pen_in_uv+Pen_in_lc+Pen_in_uc)
end

"""
Renvoie : rho_eq, rho_ineq_var, rho_ineq_cons
"""
function RhoDetail(rho::Vector,mp::NLPModels.AbstractNLPModel,nb_comp::Int64)
 nb_ineq_lvar=length(mp.meta.lvar)
 nb_ineq_uvar=length(mp.meta.uvar)
 nb_ineq_lcons=length(mp.meta.lcon)
 nb_ineq_ucons=length(mp.meta.ucon)

 return rho[1:nb_comp],rho[nb_comp+1:2*nb_comp],rho[2*nb_comp+1:2*nb_comp+nb_ineq_lvar],
        rho[2*nb_comp+nb_ineq_lvar+1:2*nb_comp+nb_ineq_lvar+nb_ineq_uvar],
        rho[2*nb_comp+nb_ineq_lvar+nb_ineq_uvar+1:2*nb_comp+nb_ineq_lvar+nb_ineq_uvar+nb_ineq_lcons],
        rho[2*nb_comp+nb_ineq_lvar+nb_ineq_uvar+nb_ineq_lcons+1:2*nb_comp+nb_ineq_lvar+nb_ineq_uvar+nb_ineq_lcons+nb_ineq_ucons]
end

#end of module
end
