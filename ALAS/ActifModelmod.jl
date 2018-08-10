module ActifModelmod

using NLPModels
import NLPModels.AbstractNLPModel, NLPModels.Counters, NLPModels.NLPModelMeta

export ActifModel, obj, grad, grad!, cons, cons!, jac_coord, jac, jprod,
       jprod!, jtprod, jtprod!, hess, hprod, hprod!

mutable struct ActifModel <: AbstractNLPModel
  meta :: NLPModelMeta

  counters :: Counters

  # Functions
  f :: Function
  g :: Function
  g! :: Function
  fg :: Function
  fg! :: Function
  H :: Function
  Hcoord :: Function
  Hp :: Function
  Hp! :: Function
  c :: Function
  c! :: Function
  fc :: Function
  fc! :: Function
  J :: Function
  Jcoord :: Function
  Jp :: Function
  Jp! :: Function
  Jtp :: Function
  Jtp! :: Function
end

NotImplemented(args...; kwargs...) = throw(NotImplementedError(""))

function ActifModel(f::Function, x0::Vector; y0::Vector = [],
    lvar::Vector = [], uvar::Vector = [], lcon::Vector = [], ucon::Vector = [],
    nnzh::Int = 0, nnzj::Int = 0,
    g::Function = NotImplemented,
    g!::Function = NotImplemented,
    fg::Function = NotImplemented,
    fg!::Function = NotImplemented,
    H::Function = NotImplemented,
    Hcoord::Function = NotImplemented,
    Hp::Function = NotImplemented,
    Hp!::Function = NotImplemented,
    c::Function = NotImplemented,
    c!::Function = NotImplemented,
    fc::Function = NotImplemented,
    fc!::Function = NotImplemented,
    J::Function = NotImplemented,
    Jcoord::Function = NotImplemented,
    Jp::Function = NotImplemented,
    Jp!::Function = NotImplemented,
    Jtp::Function = NotImplemented,
    Jtp!::Function = NotImplemented,
    name::String = "Generic",
    lin::Vector{Int} = Int[])

  nvar = length(x0)
  length(lvar) == 0 && (lvar = -Inf*ones(nvar))
  length(uvar) == 0 && (uvar =  Inf*ones(nvar))
  ncon = maximum([length(lcon); length(ucon); length(y0)])

  if ncon > 0
    length(lcon) == 0 && (lcon = -Inf*ones(ncon))
    length(ucon) == 0 && (ucon =  Inf*ones(ncon))
    length(y0) == 0   && (y0 = zeros(ncon))
  end

  nln = setdiff(1:ncon, lin)

  meta = NLPModelMeta(nvar, x0=x0, lvar=lvar, uvar=uvar, ncon=ncon, y0=y0,
    lcon=lcon, ucon=ucon, nnzj=nnzj, nnzh=nnzh, name=name, lin=lin, nln=nln)

  return ActifModel(meta, Counters(), f, g, g!, fg, fg!, H, Hcoord, Hp,
                        Hp!, c, c!, fc, fc!, J, Jcoord, Jp, Jp!, Jtp, Jtp!)
end

function obj(nlp :: ActifModel, x :: Vector)
  #increment!(nlp, :neval_obj)
  return nlp.f(x)
end

function grad(nlp :: ActifModel, x :: Vector)
  #increment!(nlp, :neval_grad)
  return nlp.g(x)
end

function grad!(nlp :: ActifModel, x :: Vector, g :: Vector)
  #increment!(nlp, :neval_grad)
  return nlp.g!(x, g)
end

function objgrad(nlp :: ActifModel, x :: Vector)
  if nlp.fg == NotImplemented
    return obj(nlp, x), grad(nlp, x)
  else
    #increment!(nlp, :neval_obj)
    #increment!(nlp, :neval_grad)
    return nlp.fg(x)
  end
end

function objgrad!(nlp :: ActifModel, x :: Vector, g :: Vector)
  if nlp.fg! == NotImplemented
    return obj(nlp, x), grad!(nlp, x, g)
  else
    #increment!(nlp, :neval_obj)
    #increment!(nlp, :neval_grad)
    return nlp.fg!(x, g)
  end
end

function cons(nlp :: ActifModel, x :: Vector)
  #increment!(nlp, :neval_cons)
  return nlp.c(x)
end

function cons!(nlp :: ActifModel, x :: Vector, c :: Vector)
  #increment!(nlp, :neval_cons)
  return nlp.c!(x, c)
end

function objcons(nlp :: ActifModel, x :: Vector)
  if nlp.fc == NotImplemented
    return obj(nlp, x), nlp.meta.ncon > 0 ? cons(nlp, x) : []
  else
    #increment!(nlp, :neval_obj)
    #increment!(nlp, :neval_cons)
    return nlp.fc(x)
  end
end

function objcons!(nlp :: ActifModel, x :: Vector, c :: Vector)
  if nlp.fc! == NotImplemented
    return obj(nlp, x), nlp.meta.ncon > 0 ? cons!(nlp, x, c) : []
  else
    #increment!(nlp, :neval_obj)
    #increment!(nlp, :neval_cons)
    return nlp.fc!(x, c)
  end
end

function jac_coord(nlp :: ActifModel, x :: Vector)
  #increment!(nlp, :neval_jac)
  return nlp.Jcoord(x)
end

function jac(nlp :: ActifModel, x :: Vector)
  #increment!(nlp, :neval_jac)
  return nlp.J(x)
end

function jprod(nlp :: ActifModel, x :: Vector, v :: Vector)
  #increment!(nlp, :neval_jprod)
  return nlp.Jp(x, v)
end

function jprod!(nlp :: ActifModel, x :: Vector, v :: Vector, Jv :: Vector)
  #increment!(nlp, :neval_jprod)
  return nlp.Jp!(x, v, Jv)
end

function jtprod(nlp :: ActifModel, x :: Vector, v :: Vector)
  #increment!(nlp, :neval_jtprod)
  return nlp.Jtp(x, v)
end

function jtprod!(nlp :: ActifModel, x :: Vector, v :: Vector, Jtv :: Vector)
  #increment!(nlp, :neval_jtprod)
  return nlp.Jtp!(x, v, Jtv)
end

function hess(nlp :: ActifModel, x :: Vector; obj_weight = 1.0,
      y :: Vector = zeros(nlp.meta.ncon))
  #increment!(nlp, :neval_hess)
  if nlp.meta.ncon > 0
    return nlp.H(x, obj_weight=obj_weight, y=y)
  else
    return nlp.H(x, obj_weight=obj_weight)
  end
end

function hess_coord(nlp :: ActifModel, x :: Vector; obj_weight = 1.0,
      y :: Vector = zeros(nlp.meta.ncon))
  #increment!(nlp, :neval_hess)
  if nlp.meta.ncon > 0
    return nlp.Hcoord(x, obj_weight=obj_weight, y=y)
  else
    return nlp.Hcoord(x, obj_weight=obj_weight)
  end
end

function hprod(nlp :: ActifModel, x :: Vector, v :: Vector;
    obj_weight = 1.0, y :: Vector = zeros(nlp.meta.ncon))
  #increment!(nlp, :neval_hprod)
  if nlp.meta.ncon > 0
    return nlp.Hp(x, v, obj_weight=obj_weight, y=y)
  else
    return nlp.Hp(x, v; obj_weight=obj_weight)
  end
end

function hprod!(nlp :: ActifModel, x :: Vector, v :: Vector, Hv :: Vector;
    obj_weight = 1.0, y :: Vector = zeros(nlp.meta.ncon))
  #increment!(nlp, :neval_hprod)
  if nlp.meta.ncon > 0
    return nlp.Hp!(x, v, Hv, obj_weight=obj_weight, y=y)
  else
    return nlp.Hp!(x, v, Hv, obj_weight=obj_weight)
  end
end

#end of module
end
