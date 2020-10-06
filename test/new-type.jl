"""
Bilevel Program:

min_{x,y} f1(x,y)
s.t.  lcon    <= c(x)   <= ucon
      lvar    <= x      <= uvar
      y \in \arg\min_{y} f2(x,y)
                 s.t. ll_lcon <= g(x,y) <= ll_ucon
                      ll_lvar <= y      <= ll_uvar

MPCC reformulation of the Bilevel Program:
       min_{x,y,lambda,mu} f1(x,y)
       s.t.  lcon    <= c(x)   <= ucon
             ll_lcon <= g(x,y) <= ll_ucon
             ll_lvar <= y      <= ll_uvar
             lvar    <= x      <= uvar
             m_lvarl <= lambda <= Inf
             m_lvarm <= mu     <= Inf
             0 = ∇f2(x,y) + ∇g(x,y)' * (lambda + mu)
             (g(x,y)  - ll_lcon) .* (lambda - m_lvar) <= 0.0
             (ll_ucon - g(x,y) ) .* (mu     - m_lvar) <= 0.0

Gradient, Jacobian and Hessian are computed using automatic differentiation
"""
mutable struct BPMPCCModel <: AbstractMPCCModel
  meta :: NLPModelMeta

  counters :: Counters

  # Functions
  f1
  f2
  c
  g

  ll_lcon
  ll_ucon
  ll_lvar
  ll_uvar
end

function BPMPCCModel(f1, x0::AbstractVector, m::AbstractVector; y0::AbstractVector = eltype(x0)[],
                    lvar::AbstractVector = eltype(x0)[], uvar::AbstractVector = eltype(x0)[],
                    lcon::AbstractVector = eltype(x0)[], ucon::AbstractVector = eltype(x0)[],
                    c = (args...)->throw(NotImplementedError("c")),
                    g = (args...)->throw(NotImplementedError("g")),
                    f2 = (args...)->throw(NotImplementedError("f2")),
                    ll_lcon::AbstractVector = eltype(x0)[], ll_ucon::AbstractVector = eltype(x0)[],
                    ll_lvar::AbstractVector = eltype(x0)[], ll_uvar::AbstractVector = eltype(x0)[],
                    name::String = "Generic", lin::AbstractVector{<: Integer}=Int[])

  nvar = length(x0) + length(m)
  length(lvar) == 0 && (lvar = -Inf*ones(nvar))
  length(uvar) == 0 && (uvar =  Inf*ones(nvar))
  ncon = maximum([length(lcon); length(ucon); length(y0)])

  meta = NLPModelMeta(nvar)

  return BPMPCCModel(meta, Counters(), f, c, G, H)
end

function JLag(f :: Function, c :: Function, x :: AbstractVector, v :: AbstractVector)
    Jtv = zeros(length(x))
    lag(x) = f(x) + dot(c(x), v)
    Jtv[1:length(x)] = ForwardDiff.gradient(lag, x)
    return Jtv
end

function HLag(f :: Function, c :: Function, x :: AbstractVector, v :: AbstractVector)
    n = length(x)
    Jtv = zeros(n,n)
    lag(x) = f(x) + dot(c(x), v)
    Jtv[1:n,1:n] = ForwardDiff.hessian(lag, x)
    return Jtv
end

function obj(nlp :: BPMPCCModel, x :: AbstractVector)
  increment!(nlp, :neval_obj)
  return nlp.f1(x)
end

function grad!(nlp :: BPMPCCModel, x :: AbstractVector, g :: AbstractVector)
  increment!(nlp, :neval_grad)
  ForwardDiff.gradient!(view(g, 1:length(x)), nlp.f1, x)
  return g
end

function cons!(nlp :: BPMPCCModel, x :: AbstractVector, c :: AbstractVector)
  increment!(nlp, :neval_cons)
  c[1:nlp.meta.ncon] = vcat(JLag(nlp.f1,nlp.c,x,v),nlp.c(x),nlp.c(x).*v)
  return c
end

function jac_structure!(nlp :: BPMPCCModel, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer})
  m, n = nlp.meta.ncon, nlp.meta.nvar
  I = ((i,j) for i = 1:m, j = 1:n)
  rows[1 : nlp.meta.nnzj] .= getindex.(I, 1)[:]
  cols[1 : nlp.meta.nnzj] .= getindex.(I, 2)[:]
  return rows, cols
end

function jac_coord!(nlp :: BPMPCCModel, x :: AbstractVector, vals :: AbstractVector)
  increment!(nlp, :neval_jac)
  Jx = ForwardDiff.jacobian(nlp.c, x)
  vals[1 : nlp.meta.nnzj] .= Jx[:]
  return vals
end

#function hess(nlp :: BPMPCCModel, x :: AbstractVector; obj_weight :: Real = one(eltype(x)))
#  increment!(nlp, :neval_hess)
#  ℓ(x) = obj_weight * nlp.f(x)
#  Hx = ForwardDiff.hessian(ℓ, x)
#  return tril(Hx)
#end

#function hess(nlp :: BPMPCCModel, x :: AbstractVector, y :: AbstractVector; obj_weight :: Real = one(eltype(x)))
#  increment!(nlp, :neval_hess)
#  ℓ(x) = obj_weight * nlp.f(x) + dot(nlp.c(x), y)
#  Hx = ForwardDiff.hessian(ℓ, x)
#  return tril(Hx)
#end

function hess_structure!(nlp :: BPMPCCModel, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer})
  n = nlp.meta.nvar
  I = ((i,j) for i = 1:n, j = 1:n if i ≥ j)
  rows[1 : nlp.meta.nnzh] .= getindex.(I, 1)
  cols[1 : nlp.meta.nnzh] .= getindex.(I, 2)
  return rows, cols
end

function hess_coord!(nlp :: BPMPCCModel, x :: AbstractVector, vals :: AbstractVector; obj_weight :: Real = one(eltype(x)))
  increment!(nlp, :neval_hess)
  ℓ(x) = obj_weight * nlp.f(x)
  Hx = ForwardDiff.hessian(ℓ, x)
  k = 1
  for j = 1 : nlp.meta.nvar
    for i = j : nlp.meta.nvar
      vals[k] = Hx[i, j]
      k += 1
    end
  end
  return vals
end

function hess_coord!(nlp :: BPMPCCModel, x :: AbstractVector, y :: AbstractVector, vals :: AbstractVector; obj_weight :: Real = one(eltype(x)))
  increment!(nlp, :neval_hess)
  ncon, ncc = nlp.meta.ncon, nlp.meta.ncc
  ℓ(x) = obj_weight * nlp.f(x) + dot(nlp.c(x), y[1:ncon]) - dot(nlp.G(x), y[ncon+1:ncon+ncc]) - dot(nlp.H(x), y[ncon+ncc+1:ncon+2*ncc])
  Hx = ForwardDiff.hessian(ℓ, x)
  k = 1
  for j = 1 : nlp.meta.nvar
    for i = j : nlp.meta.nvar
      vals[k] = Hx[i, j]
      k += 1
    end
  end
  return vals
end

function hprod!(nlp :: BPMPCCModel, x :: AbstractVector, v :: AbstractVector, Hv :: AbstractVector; obj_weight :: Real = one(eltype(x)))
  increment!(nlp, :neval_hprod)
  ℓ(x) = obj_weight * nlp.f(x)
  Hv .= ForwardDiff.derivative(t -> ForwardDiff.gradient(ℓ, x + t * v), 0)
  return Hv
end

function hprod!(nlp :: BPMPCCModel, x :: AbstractVector, y :: AbstractVector, v :: AbstractVector, Hv :: AbstractVector; obj_weight :: Real = one(eltype(x)))
  increment!(nlp, :neval_hprod)
  ncon, ncc = nlp.meta.ncon, nlp.meta.ncc
  ℓ(x) = obj_weight * nlp.f(x) + dot(nlp.c(x), y[1:ncon]) - dot(nlp.G(x), y[ncon+1:ncon+ncc]) - dot(nlp.H(x), y[ncon+ncc+1:ncon+2*ncc])
  Hv .= ForwardDiff.derivative(t -> ForwardDiff.gradient(ℓ, x + t * v), 0)
  return Hv
end
