function InitializeMPCCActif(alas::ALASMPCC,
                             xj::Vector,
                             ρ::Vector,
                             usg::Vector,
                             ush::Vector,
                             uxl::Vector,
                             uxu::Vector,
                             ucl::Vector,
                             ucu::Vector)
 #objectif du problème avec pénalité lagrangienne sans contrainte :
 pen_mpcc=CreatePenaltyNLP(alas,xj,ρ,usg,ush,uxl,uxu,ucl,ucu)

 #initialise notre problème pénalisé avec des contraintes actives w
 return ActifMPCC(pen_mpcc,alas.r,alas.s,alas.t,
                            alas.mod.nb_comp,alas.paramset,
                            alas.algoset.direction,
                            alas.algoset.linesearch)
end

function CreatePenaltyNLP(alas::ALASMPCC,
                          xj::Vector,ρ::Vector,
                          usg::Vector,ush::Vector,
                          uxl::Vector,uxu::Vector,
                          ucl::Vector,ucu::Vector)
 n=length(alas.mod.mp.meta.x0)
 nb_comp=alas.mod.nb_comp

 penf(x,yg,yh)=NLPModels.obj(alas.mod.mp,x)+Penaltygen(alas,x,yg,yh,ρ,usg,ush,uxl,uxu,ucl,ucu)
 penf(x)=penf(x[1:n],x[n+1:n+alas.mod.nb_comp],x[n+alas.mod.nb_comp+1:n+2*alas.mod.nb_comp])

 gpenf(x,yg,yh)=vec([NLPModels.grad(alas.mod.mp,x)' zeros(2*alas.mod.nb_comp)'])+GradPenaltygen(alas,x,yg,yh,ρ,usg,ush,uxl,uxu,ucl,ucu)
 gpenf(x)=gpenf(x[1:n],x[n+1:n+alas.mod.nb_comp],x[n+alas.mod.nb_comp+1:n+2*alas.mod.nb_comp])

 gfpenf(x,yg,yh)=ObjGradPenaltygen(alas,x,yg,yh,ρ,usg,ush,uxl,uxu,ucl,ucu)
 gfpenf(x)=gfpenf(x[1:n],x[n+1:n+alas.mod.nb_comp],x[n+alas.mod.nb_comp+1:n+2*alas.mod.nb_comp])

 Hpenf(x,yg,yh)=HessPenaltygen(alas,x,yg,yh,ρ,usg,ush,uxl,uxu,ucl,ucu)
 Hpenf(x)=Hpenf(x[1:n],x[n+1:n+alas.mod.nb_comp],x[n+alas.mod.nb_comp+1:n+2*alas.mod.nb_comp])


 return NLPModels.SimpleNLPModel(x->penf(x), xj,
                                lvar=[alas.mod.mp.meta.lvar;alas.tb*ones(2*alas.mod.nb_comp)], 
                                uvar=[alas.mod.mp.meta.uvar;Inf*ones(2*alas.mod.nb_comp)],
				g=x->gpenf(x),H=x->Hpenf(x),fg=x->gfpenf(x))
end

function UpdatePenaltyNLP(alas::ALASMPCC,
                          ρ::Vector,xj::Vector,
                          usg::Vector,ush::Vector,
                          uxl::Vector,uxu::Vector,
                          ucl::Vector,ucu::Vector,
                          pen_nlp::NLPModels.AbstractNLPModel;
                          gradpen::Vector=[],objpen::Float64=zeros(0),crho::Float64=1.0)
 n=length(alas.mod.mp.meta.x0)

 penf(x,yg,yh)=NLPModels.obj(alas.mod.mp,x)+Penaltygen(alas,x,yg,yh,ρ,usg,ush,uxl,uxu,ucl,ucu)
 penf(x)=penf(x[1:n],x[n+1:n+alas.mod.nb_comp],x[n+alas.mod.nb_comp+1:n+2*alas.mod.nb_comp])

 gpenf(x,yg,yh)=vec([NLPModels.grad(alas.mod.mp,x)' zeros(2*alas.mod.nb_comp)'])+GradPenaltygen(alas,x,yg,yh,ρ,usg,ush,uxl,uxu,ucl,ucu)
 gpenf(x)=gpenf(x[1:n],x[n+1:n+alas.mod.nb_comp],x[n+alas.mod.nb_comp+1:n+2*alas.mod.nb_comp])

 gfpenf(x,yg,yh)=ObjGradPenaltygen(alas,x,yg,yh,ρ,usg,ush,uxl,uxu,ucl,ucu)
 gfpenf(x)=gfpenf(x[1:n],x[n+1:n+alas.mod.nb_comp],x[n+alas.mod.nb_comp+1:n+2*alas.mod.nb_comp])

 Hpenf(x,yg,yh)=HessPenaltygen(alas,x,yg,yh,ρ,usg,ush,uxl,uxu,ucl,ucu)
 Hpenf(x)=Hpenf(x[1:n],x[n+1:n+alas.mod.nb_comp],x[n+alas.mod.nb_comp+1:n+2*alas.mod.nb_comp])

 pen_nlp.f=x->penf(x)
 pen_nlp.g=x->gpenf(x)
 pen_nlp.fg=x->gfpenf(x)
 pen_nlp.H=x->Hpenf(x)

 if isempty(objpen)
  return pen_nlp
 else
   if minimum(ρ)==maximum(ρ)
    #if minimum(rho)==alas.mod.paramset.rhomax
    #fobj=NLPModels.obj(alas.mod.mp,xj)
    #f=fobj+alas.mod.paramset.rho_update*(objpen-fobj)
    #g=alas.mod.paramset.rho_max*crho*(gradpen-vec([NLPModels.grad(alas.mod.mp,xj)' zeros(2*alas.mod.nb_comp)']))
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
function Penaltygen(alas::ALASMPCC,
                    x::Vector,yg::Vector,yh::Vector,
                    ρ::Vector,usg::Vector,ush::Vector,
                    uxl::Vector,uxu::Vector,
                    ucl::Vector,ucu::Vector)
 nb_comp=alas.mod.nb_comp

 err=viol_contrainte(alas.mod,x,yg,yh)

 return alas.algoset.penalty(alas.mod.mp,err,alas.mod.nb_comp,x,yg,yh,ρ,usg,ush,uxl,uxu,ucl,ucu)
end

function GradPenaltygen(alas::ALASMPCC,
                        x::Vector,yg::Vector,yh::Vector,
                        ρ::Vector,usg::Vector,ush::Vector,
                        uxl::Vector,uxu::Vector,
                        ucl::Vector,ucu::Vector)
 g=Vector(x)
 err=viol_contrainte(alas.mod,x,yg,yh)

 return alas.algoset.penalty(alas.mod.mp,alas.mod.G,alas.mod.H,
                                 err,alas.mod.nb_comp,
                                 x,yg,yh,ρ,
                                 usg,ush,uxl,uxu,ucl,ucu,g)
end

function HessPenaltygen(alas::ALASMPCC,
                        x::Vector,yg::Vector,yh::Vector,
                        ρ::Vector,usg::Vector,ush::Vector,
                        uxl::Vector,uxu::Vector,
                        ucl::Vector,ucu::Vector)
 n=length(alas.mod.mp.meta.x0)
 nb_comp=alas.mod.nb_comp

 Hess=zeros(0,0)
 Hf=tril([NLPModels.hess(alas.mod.mp,x) zeros(n,2*nb_comp);zeros(2*nb_comp,n) eye(2*nb_comp)])

 err=viol_contrainte(alas.mod,x,yg,yh)

 return Hf+alas.algoset.penalty(alas.mod.mp,alas.mod.G,alas.mod.H,
                                 err,nb_comp,
                                 x,yg,yh,ρ,
                                 usg,ush,uxl,uxu,ucl,ucu,Hess)
end

function ObjGradPenaltygen(alas::ALASMPCC,
                            x::Vector,yg::Vector,yh::Vector,
                            ρ::Vector,usg::Vector,ush::Vector,
                            uxl::Vector,uxu::Vector,
                            ucl::Vector,ucu::Vector)
 n=length(x)
 nb_comp=alas.mod.nb_comp

 err=viol_contrainte(alas.mod,x,yg,yh)

 f=NLPModels.obj(alas.mod.mp,x)
 f+=alas.algoset.penalty(alas.mod.mp,err,alas.mod.nb_comp,x,yg,yh,ρ,usg,ush,uxl,uxu,ucl,ucu)
 g=vec([NLPModels.grad(alas.mod.mp,x)' zeros(2*alas.mod.nb_comp)'])
 g+=alas.algoset.penalty(alas.mod.mp,alas.mod.G,alas.mod.H,
                                 err,alas.mod.nb_comp,
                                 x,yg,yh,ρ,
                                 usg,ush,uxl,uxu,ucl,ucu,Array{Float64}(0))

 return f,g
end

function ObjGradHessPenaltygen(alas::ALASMPCC,
                            x::Vector,yg::Vector,yh::Vector,
                            ρ::Vector,usg::Vector,ush::Vector,
                            uxl::Vector,uxu::Vector,
                            ucl::Vector,ucu::Vector)
 n=length(x)
 nb_comp=alas.mod.nb_comp

 err=viol_contrainte(alas.mod,x,yg,yh)

 f=NLPModels.obj(alas.mod.mp,x)
 f+=alas.algoset.penalty(alas.mod.mp,err,alas.mod.nb_comp,x,yg,yh,ρ,usg,ush,uxl,uxu,ucl,ucu)
 g=vec([NLPModels.grad(alas.mod.mp,x)' zeros(2*alas.mod.nb_comp)'])
 g+=alas.algoset.penalty(alas.mod.mp,alas.mod.G,alas.mod.H,
                                 err,alas.mod.nb_comp,
                                 x,yg,yh,ρ,
                                 usg,ush,uxl,uxu,ucl,ucu,g)
 Hess=tril([NLPModels.hess(alas.mod.mp,x) zeros(n,n);zeros(2*nb_comp,n) eye(2*nb_comp)])
 Hess+=alas.algoset.penalty(alas.mod.mp,alas.mod.G,alas.mod.H,
                                 err,nb_comp,
                                 x,yg,yh,ρ,
                                 usg,ush,uxl,uxu,ucl,ucu,Array{Float64}(0,0))

 return f,g,Hess
end
