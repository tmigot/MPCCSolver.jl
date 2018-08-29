"""
Fonction de pénalité lagrangienne
"""
function lagrangian(mp  :: AbstractNLPModel,
                    err :: Vector,
                    ncc :: Int64,
                    x   :: Vector,
                    yg  :: Vector,
                    yh  :: Vector,
                    rho :: Vector,
                    u   :: Vector)

 n=length(mp.meta.x0)

 rho_eqg,rho_eqh,rho_ineq_lvar,rho_ineq_uvar,rho_ineq_lcons,rho_ineq_ucons=_rho_detail(rho,mp,ncc)
 usg,ush,uxl,uxu,ucl,ucu = _rho_detail(u,mp,ncc)
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

function lagrangian(mp  :: AbstractNLPModel,
                    G   :: AbstractNLPModel,
                    H   :: AbstractNLPModel,
                    err :: Vector,
                    ncc :: Int64,
                    x   :: Vector,
                    yg  :: Vector,
                    yh  :: Vector,
                    rho :: Vector,
                    u   :: Vector,
                    g   :: Vector)

 n=length(mp.meta.x0)

 rho_eqg,rho_eqh,rho_ineq_lvar,rho_ineq_uvar,rho_ineq_lcons,rho_ineq_ucons=_rho_detail(rho,mp,ncc)
usg,ush,uxl,uxu,ucl,ucu = _rho_detail(u,mp,ncc)
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

function lagrangian(mp   :: AbstractNLPModel,
                    G    :: AbstractNLPModel,
                    H    :: AbstractNLPModel,
                    err  :: Vector,
                    ncc  :: Int64,
                    x    :: Vector,
                    yg   :: Vector,
                    yh   :: Vector,
                    rho  :: Vector,
                    u    :: Vector,
                    Hess :: Array{Float64,2})

 n=length(mp.meta.x0)

 rho_eqg,rho_eqh,rho_ineq_lvar,rho_ineq_uvar,rho_ineq_lcons,rho_ineq_ucons=_rho_detail(rho,mp,ncc)
usg,ush,uxl,uxu,ucl,ucu = _rho_detail(u,mp,ncc)
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
