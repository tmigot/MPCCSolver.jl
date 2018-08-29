"""
Fonction de pénalité générique :
"""
function _penalty_gen(rlx  :: RlxMPCCSolve,
                      x::Vector,yg::Vector,yh::Vector,
                      ρ::Vector,u::Vector)

 n=rlx.nlp.n
 ncc=rlx.nlp.ncc
 mod = rlx.nlp.mod

 err=viol_contrainte(mod,x,yg,yh)

 return rlx.algoset.penalty(mod.mp,err,ncc,x,yg,yh,ρ,u)
end

function _grad_penalty_gen(rlx  :: RlxMPCCSolve,
                           x::Vector,yg::Vector,yh::Vector,
                           ρ::Vector,u::Vector)

 n=rlx.nlp.n
 ncc=rlx.nlp.ncc
 mod = rlx.nlp.mod

 g=Vector(x)
 err=viol_contrainte(mod,x,yg,yh)

 return rlx.algoset.penalty(mod.mp,mod.G,mod.H,
                                 err,ncc,
                                 x,yg,yh,ρ,
                                 u,g)
end

function _hess_penalty_gen(rlx  :: RlxMPCCSolve,
                           x::Vector,yg::Vector,yh::Vector,
                           ρ::Vector,u::Vector)

 n=rlx.nlp.n
 ncc=rlx.nlp.ncc
 mod = rlx.nlp.mod

 Hess=zeros(0,0)
 Hf=tril([NLPModels.hess(mod.mp,x) zeros(n,2*ncc);zeros(2*ncc,n) zeros(2*ncc,2*ncc)])

 err=viol_contrainte(mod,x,yg,yh)

 return Hf+rlx.algoset.penalty(mod.mp,mod.G,mod.H,
                                 err,ncc,
                                 x,yg,yh,ρ,
                                 u,Hess)
end

function _objgrad_penalty_gen(rlx  :: RlxMPCCSolve,
                              x::Vector,yg::Vector,yh::Vector,
                              ρ::Vector,u::Vector)
 n=rlx.nlp.n
 ncc=rlx.nlp.ncc
 mod = rlx.nlp.mod

 err=viol_contrainte(mod,x,yg,yh)

 f=NLPModels.obj(mod.mp,x)
 f+=rlx.algoset.penalty(mod.mp,err,ncc,x,yg,yh,ρ,u)
 g=vec([NLPModels.grad(mod.mp,x)' zeros(2*ncc)'])
 g+=rlx.algoset.penalty(mod.mp,mod.G,mod.H,
                                 err,ncc,
                                 x,yg,yh,ρ,
                                 u,Array{Float64}(0))

 return f,g
end

function _objgradhess_penalty_gen(rlx  :: RlxMPCCSolve,
                                  x::Vector,yg::Vector,yh::Vector,
                                  ρ::Vector,u::Vector)
 n=rlx.nlp.n
 ncc=rlx.nlp.ncc
 mod = rlx.nlp.mod

 err=viol_contrainte(mod,x,yg,yh)

 f=NLPModels.obj(mod.mp,x)
 f+=rlx.algoset.penalty(mod.mp,err,ncc,x,yg,yh,ρ,u)
 g=vec([NLPModels.grad(mod.mp,x)' zeros(2*ncc)'])
 g+=rlx.algoset.penalty(mod.mp,mod.G,mod.H,
                                 err,ncc,
                                 x,yg,yh,ρ,
                                 u,g)
 Hess=tril([NLPModels.hess(mod.mp,x) zeros(n,n);zeros(2*ncc,n) eye(2*ncc)])
 Hess+=rlx.algoset.penalty(mod.mp,mod.G,mod.H,
                                 err,ncc,
                                 x,yg,yh,ρ,
                                 u,Array{Float64}(0,0))

 return f,g,Hess
end
