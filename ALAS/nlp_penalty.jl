function _create_penaltynlp(rlx  :: RlxMPCCSolve,
                            xj::Vector,ρ::Vector,
                            u::Vector)

 n=rlx.mod.n
 ncc=rlx.mod.ncc

 usg,ush,uxl,uxu,ucl,ucu = _rho_detail(rlx.mod,u)

 #penf(x,yg,yh)=NLPModels.obj(rlx.mod.mp,x)+_penalty_gen(rlx,x,yg,yh,ρ,usg,ush,uxl,uxu,ucl,ucu)
 penf(x,yg,yh)=obj(rlx.mod,x)+ _penalty_gen(rlx,x,yg,yh,ρ,usg,ush,uxl,uxu,ucl,ucu)
 penf(x)=penf(x[1:n],x[n+1:n+rlx.mod.ncc],x[n+rlx.mod.ncc+1:n+2*rlx.mod.ncc])

 gpenf(x,yg,yh)=vec([NLPModels.grad(rlx.mod.mp,x)' zeros(2*rlx.mod.ncc)'])+_grad_penalty_gen(rlx,x,yg,yh,ρ,usg,ush,uxl,uxu,ucl,ucu)
 #error: no matching method
 #gpenf(x,yg,yh)=vec([grad(rlx.mod,x)' zeros(2*rlx.mod.ncc)'])+_grad_penalty_gen(rlx,x,yg,yh,ρ,usg,ush,uxl,uxu,ucl,ucu)
 gpenf(x)=gpenf(x[1:n],x[n+1:n+rlx.mod.ncc],x[n+rlx.mod.ncc+1:n+2*rlx.mod.ncc])

 gfpenf(x,yg,yh)=_objgrad_penalty_gen(rlx,x,yg,yh,ρ,usg,ush,uxl,uxu,ucl,ucu)
 gfpenf(x)=gfpenf(x[1:n],x[n+1:n+rlx.mod.ncc],x[n+rlx.mod.ncc+1:n+2*rlx.mod.ncc])

 Hpenf(x,yg,yh)=_hess_penalty_gen(rlx,x,yg,yh,ρ,usg,ush,uxl,uxu,ucl,ucu)
 Hpenf(x)=Hpenf(x[1:n],x[n+1:n+rlx.mod.ncc],x[n+rlx.mod.ncc+1:n+2*rlx.mod.ncc])

 lvar = [rlx.mod.mp.meta.lvar;rlx.tb*ones(2*ncc)]
 uvar = [rlx.mod.mp.meta.uvar;Inf*ones(2*ncc)]

 penlp = NLPModels.SimpleNLPModel(x->penf(x),
                                  xj,
                                  lvar=lvar, 
                                  uvar=uvar,
				  g=x->gpenf(x),
                                  H=x->Hpenf(x),
                                  fg=x->gfpenf(x))

 return penlp
end

function _update_penaltynlp(rlx  :: RlxMPCCSolve,
                          ρ::Vector,
                          xj::Vector,
                          u::Vector,
                          pen_nlp::NLPModels.AbstractNLPModel;
                          gradpen::Vector=[],
                          objpen::Float64=zeros(0),
                          crho::Float64=1.0)

 n=rlx.mod.n
 nnbc = n+rlx.mod.ncc
 nnbc2 = n+2*rlx.mod.ncc

 usg,ush,uxl,uxu,ucl,ucu = _rho_detail(rlx.mod,u)

 penf(x,yg,yh) = NLPModels.obj(rlx.mod.mp,x) + _penalty_gen(rlx,x,yg,yh,ρ,usg,ush,uxl,uxu,ucl,ucu)
 penf(x) = penf(x[1:n],x[n+1:nnbc],x[nnbc+1:nnbc2])

 gpenf(x,yg,yh) = vec([NLPModels.grad(rlx.mod.mp,x)' zeros(2*rlx.mod.ncc)']) + _grad_penalty_gen(rlx,x,yg,yh,ρ,usg,ush,uxl,uxu,ucl,ucu)
 gpenf(x) = gpenf(x[1:n],x[n+1:nnbc],x[nnbc+1:nnbc2])

 gfpenf(x,yg,yh) = _objgrad_penalty_gen(rlx,x,yg,yh,ρ,usg,ush,uxl,uxu,ucl,ucu)
 gfpenf(x) = gfpenf(x[1:n],x[n+1:nnbc],x[nnbc+1:nnbc2])

 Hpenf(x,yg,yh) = _hess_penalty_gen(rlx,x,yg,yh,ρ,usg,ush,uxl,uxu,ucl,ucu)
 Hpenf(x) = Hpenf(x[1:n],x[n+1:nnbc],x[nnbc+1:nnbc2])

 pen_nlp.f  = x->penf(x)
 pen_nlp.g  = x->gpenf(x)
 pen_nlp.fg = x->gfpenf(x)
 pen_nlp.H  = x->Hpenf(x)

 if isempty(objpen)
  return pen_nlp
 else
   if minimum(ρ)==maximum(ρ)
    #if minimum(rho)==rlx.mod.paramset.rhomax
    #fobj=NLPModels.obj(rlx.mod.mp,xj)
    #f=fobj+rlx.mod.paramset.rho_update*(objpen-fobj)
    #g=rlx.mod.paramset.rho_max*crho*(gradpen-vec([NLPModels.grad(rlx.mod.mp,xj)' zeros(2*rlx.mod.ncc)']))
    f,g=gfpenf(xj)
   else
    f,g=gfpenf(xj)
   end

  return pen_nlp,f,g
 end
end

"""
Fonction de pénalité générique :
"""
function _penalty_gen(rlx  :: RlxMPCCSolve,
                      x::Vector,yg::Vector,yh::Vector,
                      ρ::Vector,usg::Vector,ush::Vector,
                      uxl::Vector,uxu::Vector,
                      ucl::Vector,ucu::Vector)

 ncc=rlx.mod.ncc

 err=viol_contrainte(rlx.mod,x,yg,yh)

 return rlx.algoset.penalty(rlx.mod.mp,err,rlx.mod.ncc,x,yg,yh,ρ,usg,ush,uxl,uxu,ucl,ucu)
end

function _grad_penalty_gen(rlx  :: RlxMPCCSolve,
                           x::Vector,yg::Vector,yh::Vector,
                           ρ::Vector,usg::Vector,ush::Vector,
                           uxl::Vector,uxu::Vector,
                           ucl::Vector,ucu::Vector)

 g=Vector(x)
 err=viol_contrainte(rlx.mod,x,yg,yh)

 return rlx.algoset.penalty(rlx.mod.mp,rlx.mod.G,rlx.mod.H,
                                 err,rlx.mod.ncc,
                                 x,yg,yh,ρ,
                                 usg,ush,uxl,uxu,ucl,ucu,g)
end

function _hess_penalty_gen(rlx  :: RlxMPCCSolve,
                           x::Vector,yg::Vector,yh::Vector,
                           ρ::Vector,usg::Vector,ush::Vector,
                           uxl::Vector,uxu::Vector,
                           ucl::Vector,ucu::Vector)

 n=length(rlx.mod.mp.meta.x0)
 ncc=rlx.mod.ncc

 Hess=zeros(0,0)
 Hf=tril([NLPModels.hess(rlx.mod.mp,x) zeros(n,2*ncc);zeros(2*ncc,n) eye(2*ncc)])

 err=viol_contrainte(rlx.mod,x,yg,yh)

 return Hf+rlx.algoset.penalty(rlx.mod.mp,rlx.mod.G,rlx.mod.H,
                                 err,ncc,
                                 x,yg,yh,ρ,
                                 usg,ush,uxl,uxu,ucl,ucu,Hess)
end

function _objgrad_penalty_gen(rlx  :: RlxMPCCSolve,
                              x::Vector,yg::Vector,yh::Vector,
                              ρ::Vector,usg::Vector,ush::Vector,
                              uxl::Vector,uxu::Vector,
                              ucl::Vector,ucu::Vector)
 n=length(x)
 ncc=rlx.mod.ncc

 err=viol_contrainte(rlx.mod,x,yg,yh)

 f=NLPModels.obj(rlx.mod.mp,x)
 f+=rlx.algoset.penalty(rlx.mod.mp,err,rlx.mod.ncc,x,yg,yh,ρ,usg,ush,uxl,uxu,ucl,ucu)
 g=vec([NLPModels.grad(rlx.mod.mp,x)' zeros(2*rlx.mod.ncc)'])
 g+=rlx.algoset.penalty(rlx.mod.mp,rlx.mod.G,rlx.mod.H,
                                 err,rlx.mod.ncc,
                                 x,yg,yh,ρ,
                                 usg,ush,uxl,uxu,ucl,ucu,Array{Float64}(0))

 return f,g
end

function _objgradhess_penalty_gen(rlx  :: RlxMPCCSolve,
                                  x::Vector,yg::Vector,yh::Vector,
                                  ρ::Vector,usg::Vector,ush::Vector,
                                  uxl::Vector,uxu::Vector,
                                  ucl::Vector,ucu::Vector)
 n=length(x)
 ncc=rlx.mod.ncc

 err=viol_contrainte(rlx.mod,x,yg,yh)

 f=NLPModels.obj(rlx.mod.mp,x)
 f+=rlx.algoset.penalty(rlx.mod.mp,err,rlx.mod.ncc,x,yg,yh,ρ,usg,ush,uxl,uxu,ucl,ucu)
 g=vec([NLPModels.grad(rlx.mod.mp,x)' zeros(2*rlx.mod.ncc)'])
 g+=rlx.algoset.penalty(rlx.mod.mp,rlx.mod.G,rlx.mod.H,
                                 err,rlx.mod.ncc,
                                 x,yg,yh,ρ,
                                 usg,ush,uxl,uxu,ucl,ucu,g)
 Hess=tril([NLPModels.hess(rlx.mod.mp,x) zeros(n,n);zeros(2*ncc,n) eye(2*ncc)])
 Hess+=rlx.algoset.penalty(rlx.mod.mp,rlx.mod.G,rlx.mod.H,
                                 err,ncc,
                                 x,yg,yh,ρ,
                                 usg,ush,uxl,uxu,ucl,ucu,Array{Float64}(0,0))

 return f,g,Hess
end
