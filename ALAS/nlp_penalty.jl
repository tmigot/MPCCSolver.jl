function _create_penaltynlp(rlx  :: RlxMPCCSolve,
                            xj   :: Vector,
                            ρ    :: Vector,
                            u    :: Vector)

 n   = rlx.nlp.n
 ncc = rlx.nlp.ncc
 mod = rlx.nlp.mod
 tb  = rlx.nlp.tb

 penf(x,yg,yh) = obj(rlx.nlp,x)+ _penalty_gen(rlx,x,yg,yh,ρ,u)
 penf(x) = penf(x[1:n],x[n+1:n+ncc],x[n+ncc+1:n+2*ncc])

 gpenf(x,yg,yh)=vec([RlxMPCCmod.grad(rlx.nlp,x)' zeros(2*ncc)'])+_grad_penalty_gen(rlx,x,yg,yh,ρ,u)
 #error: no matching method
 #gpenf(x,yg,yh)=vec([grad(rlx.mod,x)' zeros(2*rlx.mod.meta.ncc)'])+_grad_penalty_gen(rlx,x,yg,yh,ρ,usg,ush,uxl,uxu,ucl,ucu)
 gpenf(x)=gpenf(x[1:n],x[n+1:n+ncc],x[n+ncc+1:n+2*ncc])

 gfpenf(x,yg,yh)=_objgrad_penalty_gen(rlx,x,yg,yh,ρ,u)
 gfpenf(x)=gfpenf(x[1:n],x[n+1:n+ncc],x[n+ncc+1:n+2*ncc])

 Hpenf(x,yg,yh)=_hess_penalty_gen(rlx,x,yg,yh,ρ,u)
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
                          pen_nlp::AbstractNLPModel;
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

 penf(x,yg,yh) = obj(rlx.nlp,x) + _penalty_gen(rlx,x,yg,yh,ρ,u)
 penf(x) = penf(x[1:n],x[n+1:nnbc],x[nnbc+1:nnbc2])

 gpenf(x,yg,yh) = vec([RlxMPCCmod.grad(rlx.nlp,x)' zeros(2*ncc)']) + _grad_penalty_gen(rlx,x,yg,yh,ρ,u)
 gpenf(x) = gpenf(x[1:n],x[n+1:nnbc],x[nnbc+1:nnbc2])

 gfpenf(x,yg,yh) = _objgrad_penalty_gen(rlx,x,yg,yh,ρ,u)
 gfpenf(x) = gfpenf(x[1:n],x[n+1:nnbc],x[nnbc+1:nnbc2])

 Hpenf(x,yg,yh) = _hess_penalty_gen(rlx,x,yg,yh,ρ,u)
 Hpenf(x) = Hpenf(x[1:n],x[n+1:nnbc],x[nnbc+1:nnbc2])

 pen_nlp.f  = x->penf(x)
 pen_nlp.g  = x->gpenf(x)
 pen_nlp.fg = x->gfpenf(x)
 pen_nlp.H  = x->Hpenf(x)

###################################
# A REVOIR
 if isempty(objpen)
  return pen_nlp
 else
   if minimum(ρ)==maximum(ρ)
    f,g=gfpenf(xj)
   else
    f,g=gfpenf(xj)
   end

  return pen_nlp,f,g
 end
###################################
end

#################################################################
#
# Fonctions de pénalité générique
#
#################################################################

include("nlp_penalty_gen.jl")
