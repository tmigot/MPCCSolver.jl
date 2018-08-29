##########################################################################
#
# P(hx,bx,gx) = 1/2 * (||h(x)||^2+||b(x)||^2+||g(x)||^2)
#
##########################################################################
function quadratic(hx  :: Vector,
                   bx  :: Vector,
                   gx  :: Vector,
                   rho :: Vector,
                   u   :: Vector)

 lh, lb, lg = length(hx), length(bx), length(gx)
 rhoh, rhob, rhog = rho[1:lh], rho[lh+1:lh+lb], rho[lh+lb+1:lh+lb+lg]

 return 0.5*(dot(rhoh.*hx,hx)+dot(rhog.*gx,gx)+dot(rhob.*bx,bx))
end

##########################################################################
#
# \nabla P(hx,bx,gx)
#
##########################################################################
function quadratic(hx    :: Vector,
                   bx    :: Vector,
                   gx    :: Vector,
                   grad  :: Vector,
                   rho   :: Vector,
                   u     :: Vector)

 lh, lb, lg = length(hx), length(bx), length(gx)
 rhoh, rhob, rhog = rho[1:lh], rho[lh+1:lh+lb], rho[lh+lb+1:lh+lb+lg]

 grad = vcat(vcat(hx,bx,gx).*rho,hx.*rhoh)

 return grad
end

##########################################################################
#
# \nabla P(hx,bx,gx), \nabla^2 P(hx,bx,gx)
#
##########################################################################
function quadratic(hx    :: Vector,
                   bx    :: Vector,
                   gx    :: Vector,
                   grad  :: Vector,
                   Hx    :: Any, #matrix type
                   rho   :: Vector,
                   u     :: Vector)

 lh, lb, lg = length(hx), length(bx), length(gx)
 rhoh, rhob, rhog = rho[1:lh], rho[lh+1:lh+lb], rho[lh+lb+1:lh+lb+lg]

 grad = vcat(vcat(hx,bx,gx).*rho,hx.*rhoh)
 Hx = diagm(rho)

 return grad, Hx
end


function quadratic(mp  :: AbstractNLPModel,
                   err :: Vector,
                   ncc :: Int64,
                   x   :: Vector,
                   yg  :: Vector,
                   yh  :: Vector,
                   rho :: Vector,
                   u   :: Vector)

 n = length(mp.meta.x0)

 rho_eqg,rho_eqh,rho_ineq_lvar,rho_ineq_uvar,rho_ineq_lcons,rho_ineq_ucons=_rho_detail(rho,mp,ncc)
 usg,ush,uxl,uxu,ucl,ucu = _rho_detail(u,mp,ncc)
 err_eq_g,err_eq_h,err_in_lv,err_in_uv,err_in_lc,err_in_uc=_err_detail(err,n,ncc,mp.meta.ncon)

 if ncc>0
  Pen_eq = dot(rho_eqg.*err_eq_g,err_eq_g)+dot(rho_eqh.*err_eq_h,err_eq_h)
 else
  Pen_eq = 0.0
 end

 Pen_in_lv = dot(rho_ineq_lvar.*err_in_lv,err_in_lv)
 Pen_in_uv = dot(rho_ineq_uvar.*err_in_uv,err_in_uv)

 if mp.meta.ncon != 0

  Pen_in_lc = dot(rho_ineq_lcons.*err_in_lc,err_in_lc)
  Pen_in_uc = dot(rho_ineq_ucons.*err_in_uc,err_in_uc)

 else

  Pen_in_uc = 0.0
  Pen_in_lc = 0.0

 end

 return 0.5*(Pen_eq + Pen_in_lv + Pen_in_uv + Pen_in_lc + Pen_in_uc)
end

function quadratic(mp         :: AbstractNLPModel,
                   err        :: Vector,
                   ncc        :: Int64,
                   x          :: Vector,
                   yg         :: Vector,
                   yh         :: Vector,
                   rho        :: Vector,
                   u          :: Vector,
                   f          :: Float64,
                   rho_update :: Float64)

 return rho_update*f
end

function quadratic(mp  :: AbstractNLPModel,
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

  G_in_lv = -rho_ineq_lvar.*err_in_lv
  G_in_uv =  rho_ineq_uvar.*err_in_uv

  if mp.meta.ncon != 0

   Jtv(z) = jtprod(mp,x,z)

   G_in_lc = -Jtv(rho_ineq_lcons.*err_in_lc)
   G_in_uc =  Jtv(rho_ineq_ucons.*err_in_uc)

  else

   G_in_uc = zeros(n)
   G_in_lc = zeros(n)

  end

  if ncc > 0 
   G_eq = jtprod(G,x,rho_eqg.*err_eq_g) + jtprod(H,x,rho_eqh.*err_eq_h)
   return vec([(G_eq+G_in_lv+G_in_uv+G_in_lc+G_in_uc)' (-rho_eqg.*err_eq_g)' (-rho_eqh.*err_eq_h)'])
  else
   return G_in_lv+G_in_uv+G_in_lc+G_in_uc
  end
end

function quadratic(mp         :: AbstractNLPModel,
                   G          :: AbstractNLPModel,
                   H          :: AbstractNLPModel,
                   err        :: Vector,
                   ncc        :: Int64,
                   x          :: Vector,
                   yg         :: Vector,
                   yh         :: Vector,
                   rho        :: Vector,
                   u          :: Vector,
                   g          :: Vector,
                   rho_update :: Float64)

 return rho_update*g
end

function quadratic(mp   :: AbstractNLPModel,
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

   ilc = find(x->x>0,err_in_lc);rho_ineq_lcons[ilc]=zeros(length(ilc))
   iuc = find(x->x>0,err_in_uc);rho_ineq_ucons[iuc]=zeros(length(iuc))

   J = jac(mp,x)
   Hlc = hess(mp, x, obj_weight=1.0, y=rho_ineq_lcons.*err_in_lc) + (diagm(rho_ineq_lcons)*J)'*J
   Huc = hess(mp, x, obj_weight=1.0, y=rho_ineq_ucons.*err_in_uc) + (diagm(rho_ineq_ucons)*J)'*J

   JG = jac(G,x)
   HCG=hess(G, x, obj_weight=0.0, y=err_eq_g)+(diagm(rho_eqg)*JG)'*JG

   JH = jac(H,x)
   HCH=hess(H, x, obj_weight=0.0, y=err_eq_h)+(diagm(rho_eqh)*JH)'*JH

   return tril([diagm(rho_ineq_lvar.*lv)+diagm(rho_ineq_uvar.*uv)+HCG+HCH+Hlc+Huc zeros(n,2*ncc);zeros(2*ncc,n) diagm([rho_eqg;rho_eqh])])
  elseif ncc>0
   JH = jac(H,x)
   JG = jac(G,x)
   HCG=hess(G, x, obj_weight=0.0, y=err_eq_g)+(diagm(rho_eqg)*JG)'*JG
   HCH=hess(H, x, obj_weight=0.0, y=err_eq_h)+(diagm(rho_eqh)*JH)'*JH

   return tril([diagm(rho_ineq_lvar.*lv)+diagm(rho_ineq_uvar.*uv)+HCG+HCH zeros(n,2*ncc);zeros(2*ncc,n) diagm([rho_eqg;rho_eqh])])
  elseif mp.meta.ncon!=0
   ilc=find(x->x>0,err_in_lc);rho_ineq_lcons[ilc]=zeros(length(ilc))
   iuc=find(x->x>0,err_in_uc);rho_ineq_ucons[iuc]=zeros(length(iuc))

   J = jac(mp,x)
   Hlc=hess(mp, x, obj_weight=1.0, y=rho_ineq_lcons.*err_in_lc)+(diagm(rho_ineq_lcons)*J)'*J
   Huc=hess(mp, x, obj_weight=1.0, y=rho_ineq_ucons.*err_in_uc)+(diagm(rho_ineq_ucons)*J)'*J
   return tril(diagm(rho_ineq_lvar.*lv)+diagm(rho_ineq_uvar.*uv)+Hlc+Huc)
  else
   return tril(diagm(rho_ineq_lvar.*lv)+diagm(rho_ineq_uvar.*uv))
  end
end

function quadratic(mp         :: AbstractNLPModel,
                   G          :: AbstractNLPModel,
                   H          :: AbstractNLPModel,
                   err        :: Vector,
                   ncc        :: Int64,
                   x          :: Vector,
                   yg         :: Vector,
                   yh         :: Vector,
                   rho        :: Vector,
                   u          :: Vector,
                   Hess       :: Array{Float64,2},
                   rho_update :: Float64)

 return rho_update*Hess
end
