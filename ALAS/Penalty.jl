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
using ForwardDiff

"""
Fonction de pénalité quadratique
"""
function Quadratic(mp::NLPModels.AbstractNLPModel,G::Function,H::Function,nb_comp::Int64,x::Vector,yg::Vector,
                   yh::Vector,rho::Vector,usg::Vector,ush::Vector,uxl::Vector,uxu::Vector,ucl::Vector,ucu::Vector)
 
 f=NLPModels.obj(mp,x)

 #mod::MPCCmod.MPCC,r::Float64,s::Float64,t::Float64
 rho_eqg,rho_eqh,rho_ineq_lvar,rho_ineq_uvar,rho_ineq_lcons,rho_ineq_ucons=RhoDetail(rho,mp,nb_comp)

# Pen_eq2=rhof*norm((alas.mod.G(x)-yg))^2+rhof*norm((alas.mod.H(x)-yh))^2
if nb_comp>0
 err_eq_g=(G(x)-yg)
 err_eq_h=(H(x)-yh)
 Pen_eq=dot(rho_eqg.*err_eq_g,err_eq_g)+dot(rho_eqh.*err_eq_h,err_eq_h)
else
 Pen_eq=0
end

# Pen_in_lv=norm(sqrt(rho_ineq_lvar).*max(mod.mp.meta.lvar-x+uxl./rho_ineq_lvar,0.0))^2
 err_in_lv=max(mp.meta.lvar-x,zeros(length(mp.meta.lvar)))
 Pen_in_lv=dot(rho_ineq_lvar.*err_in_lv,err_in_lv)
# Pen_in_uv=norm(sqrt(rho_ineq_uvar).*max(x-mod.mp.meta.uvar+uxu./rho_ineq_uvar,0.0))^2
 err_in_uv=max(x-mp.meta.uvar,zeros(length(mp.meta.uvar)))
 Pen_in_uv=dot(rho_ineq_uvar.*err_in_uv,err_in_uv)
# Pen_in_lc=norm(sqrt(rho_ineq_lcons).*max(mod.mp.meta.lcon-alas.mod.mp.c(x)+ucl./rho_ineq_lcons,0.0))^2
 if mp.meta.ncon!=0
  c=NLPModels.cons(mp,x)
  err_in_lc=max(mp.meta.lcon-c,zeros(length(mp.meta.lcon)))
  Pen_in_lc=dot(rho_ineq_lcons.*err_in_lc,err_in_lc)
  err_in_uc=max(c-mp.meta.ucon,zeros(length(mp.meta.ucon)))
  Pen_in_uc=dot(rho_ineq_ucons.*err_in_uc,err_in_uc)
 else
  Pen_in_uc=0
  Pen_in_lc=0
 end

 return f+0.5*(Pen_eq+Pen_in_lv+Pen_in_uv+Pen_in_lc+Pen_in_uc)
end

function Quadratic(mp::NLPModels.AbstractNLPModel,G::NLPModels.AbstractNLPModel,H::NLPModels.AbstractNLPModel,nb_comp::Int64,x::Vector,yg::Vector,yh::Vector,rho::Vector,usg::Vector,ush::Vector,uxl::Vector,uxu::Vector,ucl::Vector,ucu::Vector,dev::String)
 
 n=length(mp.meta.x0)

 rho_eqg,rho_eqh,rho_ineq_lvar,rho_ineq_uvar,rho_ineq_lcons,rho_ineq_ucons=RhoDetail(rho,mp,nb_comp)

 if nb_comp>0
  err_eq_g=(NLPModels.cons(G,x)-yg)
  err_eq_h=(NLPModels.cons(H,x)-yh)
 end

 err_in_lv=max(mp.meta.lvar-x,zeros(length(mp.meta.lvar)))
 err_in_uv=max(x-mp.meta.uvar,zeros(length(mp.meta.uvar)))

 if mp.meta.ncon!=0
  c=NLPModels.cons(mp,x)

  err_in_lc=max(mp.meta.lcon-c,zeros(length(mp.meta.lcon)))
  err_in_uc=max(c-mp.meta.ucon,zeros(length(mp.meta.ucon)))
 end

 if dev=="grad"
  G_in_lv=-rho_ineq_lvar.*err_in_lv
  G_in_uv=rho_ineq_uvar.*err_in_uv
  if mp.meta.ncon!=0
   Jtv(z)=NLPModels.jtprod(mp,x,z)
   G_in_lc=-Jtv(rho_ineq_lcons.*err_in_lc)
   G_in_uc=Jtv(rho_ineq_ucons.*err_in_uc)
  else
   G_in_uc=zeros(n)
   G_in_lc=zeros(n)
  end
  if nb_comp>0
   G_eq=NLPModels.jtprod(G,x,rho_eqg.*err_eq_g)+NLPModels.jtprod(H,x,rho_eqh.*err_eq_h)

   return vec([(NLPModels.grad(mp,x)+G_eq+G_in_lv+G_in_uv+G_in_lc+G_in_uc)' -rho_eqg.*err_eq_g -rho_eqh.*err_eq_h])
  else
   return NLPModels.grad(mp,x)+G_in_lv+G_in_uv+G_in_lc+G_in_uc
  end
 elseif dev=="hess"
  lv=zeros(n);ilv=find(x->x>0,mp.meta.lvar-x);lv[ilv]=ones(length(ilv));
  uv=zeros(n);iuv=find(x->x>0,x-mp.meta.uvar);uv[iuv]=ones(length(iuv));
  if nb_comp>0 && mp.meta.ncon!=0
   ilc=find(x->x>0,mp.meta.lcon-c);rho_ineq_lcons[ilc]=zeros(length(ilc))
   iuc=find(x->x>0,c-mp.meta.ucon);rho_ineq_ucons[iuc]=zeros(length(iuc))
   Hlc=hess(mp, x, obj_weight=1.0, y=rho_ineq_lcons.*err_in_lc)+(diagm(rho_ineq_lcons)*NLPModels.jac(mp,x))'*NLPModels.jac(mp,x)
   Huc=hess(mp, x, obj_weight=1.0, y=rho_ineq_ucons.*err_in_uc)+(diagm(rho_ineq_ucons)*NLPModels.jac(mp,x))'*NLPModels.jac(mp,x)

   HCG=hess(G, x, obj_weight=1.0, y=err_eq_g)+(diagm(rho_eqg)*NLPModels.jac(G,x))'*NLPModels.jac(G,x)
   HCH=hess(H, x, obj_weight=1.0, y=err_eq_h)+(diagm(rho_eqh)*NLPModels.jac(H,x))'*NLPModels.jac(H,x)

   return tril([NLPModels.hess(mp,x)+diagm(rho_ineq_lvar.*lv)+diagm(rho_ineq_uvar.*uv)+HCG+HCH+Hlc+Huc zeros(n,2*nb_comp);zeros(2*nb_comp,n) diagm([rho_eqg;rho_eqh])])
  elseif nb_comp>0
   HCG=hess(G, x, obj_weight=1.0, y=err_eq_g)+(diagm(rho_eqg)*NLPModels.jac(G,x))'*NLPModels.jac(G,x)
   HCH=hess(H, x, obj_weight=1.0, y=err_eq_h)+(diagm(rho_eqh)*NLPModels.jac(H,x))'*NLPModels.jac(H,x)
   return tril([NLPModels.hess(mp,x)+diagm(rho_ineq_lvar.*lv)+diagm(rho_ineq_uvar.*uv)+HCG+HCH zeros(n,n);zeros(2*nb_comp,n) diagm([rho_eqg;rho_eqh])])
  elseif mp.meta.ncon!=0
   ilc=find(x->x>0,mp.meta.lcon-c);rho_ineq_lcons[ilc]=zeros(length(ilc))
   iuc=find(x->x>0,c-mp.meta.ucon);rho_ineq_ucons[iuc]=zeros(length(iuc))
   Hlc=hess(mp, x, obj_weight=1.0, y=rho_ineq_lcons.*err_in_lc)+(diagm(rho_ineq_lcons)*NLPModels.jac(mp,x))'*NLPModels.jac(mp,x)
   Huc=hess(mp, x, obj_weight=1.0, y=rho_ineq_ucons.*err_in_uc)+(diagm(rho_ineq_ucons)*NLPModels.jac(mp,x))'*NLPModels.jac(mp,x)
   return tril(NLPModels.hess(mp,x)+diagm(rho_ineq_lvar.*lv)+diagm(rho_ineq_uvar.*uv)+Hlc+Huc)
  else
   return tril(NLPModels.hess(mp,x)+diagm(rho_ineq_lvar.*lv)+diagm(rho_ineq_uvar.*uv))
  end
 end
end


"""
Fonction de pénalité lagrangienne
"""
function Lagrangian(mp::NLPModels.AbstractNLPModel,G::Function,H::Function,nb_comp::Int64,x::Vector,yg::Vector,yh::Vector,rho::Vector,usg::Vector,ush::Vector,uxl::Vector,uxu::Vector,ucl::Vector,ucu::Vector)
 f(z)=NLPModels.obj(mp,z)
 c(z)=NLPModels.cons(mp,z)

 #mod::MPCCmod.MPCC,r::Float64,s::Float64,t::Float64
 rho_eqg,rho_eqh,rho_ineq_lvar,rho_ineq_uvar,rho_ineq_lcons,rho_ineq_ucons=RhoDetail(rho,mp,nb_comp)

 Lagrangian=dot(G(x)-yg,usg)+dot(H(x)-yh,ush)
# Pen_eq2=rhof*norm((G(x)-yg))^2+rhof*norm((H(x)-yh))^2
 err_eq_g=(G(x)-yg)
 err_eq_h=(H(x)-yh)
 Pen_eq=dot(rho_eqg.*err_eq_g,err_eq_g)+dot(rho_eqh.*err_eq_h,err_eq_h)
# Pen_in_lv=norm(sqrt(rho_ineq_lvar).*max(mp.meta.lvar-x+uxl./rho_ineq_lvar,0.0))^2
 err_in_lv=max(mp.meta.lvar-x+uxl./rho_ineq_lvar,0.0)
 Pen_in_lv=dot(rho_ineq_lvar.*err_in_lv,err_in_lv)
# Pen_in_uv=norm(sqrt(rho_ineq_uvar).*max(x-mp.meta.uvar+uxu./rho_ineq_uvar,0.0))^2
 err_in_uv=max(x-mp.meta.uvar+uxu./rho_ineq_uvar,0.0)
 Pen_in_uv=dot(rho_ineq_uvar.*err_in_uv,err_in_uv)
# Pen_in_lc=norm(sqrt(rho_ineq_lcons).*max(mp.meta.lcon-mp.c(x)+ucl./rho_ineq_lcons,0.0))^2
 err_in_lc=max(mp.meta.lcon-c(x)+ucl./rho_ineq_lcons,0.0)
 Pen_in_lc=dot(rho_ineq_lcons.*err_in_lc,err_in_lc)
 err_in_uc=max(c(x)-mp.meta.ucon+ucu./rho_ineq_ucons,0.0)
 Pen_in_uc=dot(rho_ineq_ucons.*err_in_uc,err_in_uc)

 return f(x)+Lagrangian+0.5*(Pen_eq+Pen_in_lv+Pen_in_uv+Pen_in_lc+Pen_in_uc)
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
