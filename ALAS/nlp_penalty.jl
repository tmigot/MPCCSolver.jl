function _create_penaltynlp(rlx  :: RlxMPCCSolve,
                            xj::Vector,ρ::Vector,
                            u::Vector)

 n=rlx.nlp.n
 ncc=rlx.nlp.ncc
 mod = rlx.nlp.mod
 tb  = rlx.nlp.tb

 usg,ush,uxl,uxu,ucl,ucu = _rho_detail(mod,u)

 #penf(x,yg,yh)=NLPModels.obj(rlx.mod.mp,x)+_penalty_gen(rlx,x,yg,yh,ρ,usg,ush,uxl,uxu,ucl,ucu)
 penf(x,yg,yh)=obj(mod,x)+ _penalty_gen(rlx,x,yg,yh,ρ,usg,ush,uxl,uxu,ucl,ucu)
 penf(x)=penf(x[1:n],x[n+1:n+ncc],x[n+ncc+1:n+2*ncc])

 gpenf(x,yg,yh)=vec([NLPModels.grad(mod.mp,x)' zeros(2*ncc)'])+_grad_penalty_gen(rlx,x,yg,yh,ρ,usg,ush,uxl,uxu,ucl,ucu)
 #error: no matching method
 #gpenf(x,yg,yh)=vec([grad(rlx.mod,x)' zeros(2*rlx.mod.ncc)'])+_grad_penalty_gen(rlx,x,yg,yh,ρ,usg,ush,uxl,uxu,ucl,ucu)
 gpenf(x)=gpenf(x[1:n],x[n+1:n+ncc],x[n+ncc+1:n+2*ncc])

 gfpenf(x,yg,yh)=_objgrad_penalty_gen(rlx,x,yg,yh,ρ,usg,ush,uxl,uxu,ucl,ucu)
 gfpenf(x)=gfpenf(x[1:n],x[n+1:n+ncc],x[n+ncc+1:n+2*ncc])

 Hpenf(x,yg,yh)=_hess_penalty_gen(rlx,x,yg,yh,ρ,usg,ush,uxl,uxu,ucl,ucu)
 Hpenf(x)=Hpenf(x[1:n],x[n+1:n+ncc],x[n+ncc+1:n+2*ncc])

 lvar = [mod.mp.meta.lvar;tb*ones(2*ncc)]
 uvar = [mod.mp.meta.uvar;Inf*ones(2*ncc)]

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

 n=rlx.nlp.n
 ncc=rlx.nlp.ncc
 mod = rlx.nlp.mod
 tb  = rlx.nlp.tb

 nnbc = n+ncc
 nnbc2 = n+2*ncc

 usg,ush,uxl,uxu,ucl,ucu = _rho_detail(mod,u)

 penf(x,yg,yh) = NLPModels.obj(mod.mp,x) + _penalty_gen(rlx,x,yg,yh,ρ,usg,ush,uxl,uxu,ucl,ucu)
 penf(x) = penf(x[1:n],x[n+1:nnbc],x[nnbc+1:nnbc2])

 gpenf(x,yg,yh) = vec([NLPModels.grad(mod.mp,x)' zeros(2*ncc)']) + _grad_penalty_gen(rlx,x,yg,yh,ρ,usg,ush,uxl,uxu,ucl,ucu)
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
    #fobj=NLPModels.obj(mod.mp,xj)
    #f=fobj+rlx.paramset.rho_update*(objpen-fobj)
    #g=rlx.paramset.rho_max*crho*(gradpen-vec([NLPModels.grad(mod.mp,xj)' zeros(2*ncc)']))
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

 n=rlx.nlp.n
 ncc=rlx.nlp.ncc
 mod = rlx.nlp.mod

 err=viol_contrainte(mod,x,yg,yh)

 return rlx.algoset.penalty(mod.mp,err,ncc,x,yg,yh,ρ,usg,ush,uxl,uxu,ucl,ucu)
end

function _grad_penalty_gen(rlx  :: RlxMPCCSolve,
                           x::Vector,yg::Vector,yh::Vector,
                           ρ::Vector,usg::Vector,ush::Vector,
                           uxl::Vector,uxu::Vector,
                           ucl::Vector,ucu::Vector)

 n=rlx.nlp.n
 ncc=rlx.nlp.ncc
 mod = rlx.nlp.mod

 g=Vector(x)
 err=viol_contrainte(mod,x,yg,yh)

 return rlx.algoset.penalty(mod.mp,mod.G,mod.H,
                                 err,ncc,
                                 x,yg,yh,ρ,
                                 usg,ush,uxl,uxu,ucl,ucu,g)
end

function _hess_penalty_gen(rlx  :: RlxMPCCSolve,
                           x::Vector,yg::Vector,yh::Vector,
                           ρ::Vector,usg::Vector,ush::Vector,
                           uxl::Vector,uxu::Vector,
                           ucl::Vector,ucu::Vector)

 n=rlx.nlp.n
 ncc=rlx.nlp.ncc
 mod = rlx.nlp.mod

 Hess=zeros(0,0)
 Hf=tril([NLPModels.hess(mod.mp,x) zeros(n,2*ncc);zeros(2*ncc,n) eye(2*ncc)])

 err=viol_contrainte(mod,x,yg,yh)

 return Hf+rlx.algoset.penalty(mod.mp,mod.G,mod.H,
                                 err,ncc,
                                 x,yg,yh,ρ,
                                 usg,ush,uxl,uxu,ucl,ucu,Hess)
end

function _objgrad_penalty_gen(rlx  :: RlxMPCCSolve,
                              x::Vector,yg::Vector,yh::Vector,
                              ρ::Vector,usg::Vector,ush::Vector,
                              uxl::Vector,uxu::Vector,
                              ucl::Vector,ucu::Vector)
 n=rlx.nlp.n
 ncc=rlx.nlp.ncc
 mod = rlx.nlp.mod

 err=viol_contrainte(mod,x,yg,yh)

 f=NLPModels.obj(mod.mp,x)
 f+=rlx.algoset.penalty(mod.mp,err,ncc,x,yg,yh,ρ,usg,ush,uxl,uxu,ucl,ucu)
 g=vec([NLPModels.grad(mod.mp,x)' zeros(2*ncc)'])
 g+=rlx.algoset.penalty(mod.mp,mod.G,mod.H,
                                 err,ncc,
                                 x,yg,yh,ρ,
                                 usg,ush,uxl,uxu,ucl,ucu,Array{Float64}(0))

 return f,g
end

function _objgradhess_penalty_gen(rlx  :: RlxMPCCSolve,
                                  x::Vector,yg::Vector,yh::Vector,
                                  ρ::Vector,usg::Vector,ush::Vector,
                                  uxl::Vector,uxu::Vector,
                                  ucl::Vector,ucu::Vector)
 n=rlx.nlp.n
 ncc=rlx.nlp.ncc
 mod = rlx.nlp.mod

 err=viol_contrainte(mod,x,yg,yh)

 f=NLPModels.obj(mod.mp,x)
 f+=rlx.algoset.penalty(mod.mp,err,ncc,x,yg,yh,ρ,usg,ush,uxl,uxu,ucl,ucu)
 g=vec([NLPModels.grad(mod.mp,x)' zeros(2*ncc)'])
 g+=rlx.algoset.penalty(mod.mp,mod.G,mod.H,
                                 err,ncc,
                                 x,yg,yh,ρ,
                                 usg,ush,uxl,uxu,ucl,ucu,g)
 Hess=tril([NLPModels.hess(mod.mp,x) zeros(n,n);zeros(2*ncc,n) eye(2*ncc)])
 Hess+=rlx.algoset.penalty(mod.mp,mod.G,mod.H,
                                 err,ncc,
                                 x,yg,yh,ρ,
                                 usg,ush,uxl,uxu,ucl,ucu,Array{Float64}(0,0))

 return f,g,Hess
end
