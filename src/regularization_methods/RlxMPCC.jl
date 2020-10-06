import NLPModels: obj, grad, grad!, objgrad, objgrad!, objcons, objcons!, cons, cons!,
                  jth_con, jth_congrad, jth_congrad!, jth_sparse_congrad,
                  jac_structure!, jac_structure, jac_coord!, jac_coord,
                  jac, jprod, jprod!, jtprod, jtprod!, jac_op, jac_op!,
                  jth_hprod, jth_hprod!, ghjvprod, ghjvprod!,
                  hess_structure!, hess_structure, hess_coord!, hess_coord,
                  hess, hess_op, hess_op!, hprod, hprod!

"""
Convert an MPCCModel to a parametric NLPModels as follows.
Definit le type RlxMPCC :
min 	f(x)
s.t. 	l <= x <= u
	 lcon <= cnl(x,t,tb) <= ucon

with

cnl(x) := c(x),G(x)+tb,H(x)+tb,Phi(G(x).*H(x),t)
"""
mutable struct RlxMPCC <: AbstractNLPModel

 meta     :: NLPModelMeta
 counters :: Counters
 mod      :: AbstractMPCCModel

 r        :: Float64
 s        :: Float64
 t        :: Float64
 tb       :: Float64

 function RlxMPCC(mod    :: AbstractMPCCModel,
                  r      :: Float64,
                  s      :: Float64,
                  t      :: Float64,
                  tb     :: Float64;
				  name   :: String = "Generic",
				  lin    :: AbstractVector{<: Integer}=Int[])
  x = mod.meta.x0

  nvar, ncc = mod.meta.nvar, mod.meta.ncc
  new_lcon = vcat(mod.meta.lcon,
                  mod.meta.lccG,
                  mod.meta.lccH,
                  -Inf*ones(ncc))
  new_ucon = vcat(mod.meta.ucon,
                  Inf*ones(2*ncc),
                  zeros(ncc))
  ncon = maximum([length(new_lcon); length(new_ucon)])
  y0 = vcat(mod.meta.y0, zeros(3*ncc))

  nnzh = nvar * (nvar + 1) / 2
  nnzj = nvar * ncon
  nln = setdiff(1:ncon, lin)

  meta = NLPModelMeta(nvar, x0 = x, lvar = mod.meta.lvar, uvar = mod.meta.uvar,
                                    ncon = mod.meta.ncon+3*ncc, y0=y0,
                                    lcon = new_lcon, ucon = new_ucon,
								    nnzj=nnzj, nnzh=nnzh, lin=lin, nln=nln,
								    minimize=true, islp=false, name=name)

  if minimum([r,s,t,tb])<0.0
   throw(error("Domain error: (r,s,t) must be non-negative"))
  end

  return new(meta, Counters(), mod, r, s, t, tb)
 end
end

"""
Return the number of classical nonlinear constraints
"""
function get_ncon(rlxmpcc :: RlxMPCC) return rlxmpcc.meta.ncon - 3*rlxmpcc.ncc end

"""
update_rlx!(:: RlxMPCC, :: Float64, :: Float64, :: Float64, :: Float64)
update the parameters r,s,t,tb of the RlxMPCC.
"""
function update_rlx!(rlx :: RlxMPCC, r :: Float64, s :: Float64, t :: Float64, tb :: Float64)
 rlx.r, rlx.s, rlx.t, rlx.tb = r, s, t, tb
 return rlx
end

function obj(nlp :: RlxMPCC, x ::  AbstractVector)
 increment!(nlp, :neval_obj)
 return obj(nlp.mod, x)
end

function grad!(nlp :: RlxMPCC, x :: Vector, gx :: AbstractVector)
 increment!(nlp, :neval_grad)
 return grad!(nlp.mod, x, gx)
end

function cons!(nlp :: RlxMPCC, x :: AbstractVector, c :: AbstractVector)
  increment!(nlp, :neval_cons)
  if nlp.mod.meta.ncc > 0
   Gx, Hx = consG(nlp.mod, x), consH(nlp.mod, x)
   c[1:nlp.meta.ncon] .= vcat(cons_nl(nlp.mod, x),
                              Gx .+ nlp.tb,
	  						  Hx .+ nlp.tb,
							  phi(Gx, Hx, nlp.r, nlp.s, nlp.t))
  else
   c[1:nlp.meta.ncon] .= cons_nl(nlp.mod, x)
  end

  return c
end

function jac(nlp :: RlxMPCC, x :: AbstractVector)
 increment!(nlp, :neval_jac)
 ncc, mod = nlp.mod.meta.ncc, nlp.mod

 if nlp.mod.meta.ncc > 0
  JGx, JHx = jacG(mod,x),   jacH(mod,x)
  Gx,  Hx  = consG(mod, x), consH(mod, x)
  dp = dphi(Gx, Hx, nlp.r, nlp.s, nlp.t)
  A = vcat(jac_nl(mod,x),
           JGx, JHx,
           diagm(0 => dp[1:ncc]) * JGx + diagm(0 => dp[ncc+1:2*ncc]) * JHx)
  else
   A = jac_nl(mod,x)
  end

 return A
end

function jac_coord!(nlp :: RlxMPCC, x :: AbstractVector, vals ::AbstractVector)
 Jx = jac(nlp, x) #findnz(jac(nlp.mod, x))
 m, n = size(Jx)
 vals[1 : n*m] .= Jx[:]
 return vals
end

function jac_structure!(nlp :: RlxMPCC,
	                   rows :: AbstractVector{<:Integer},
					   cols :: AbstractVector{<:Integer})
 m, n = nlp.meta.ncon, nlp.meta.nvar
 I = ((i,j) for i = 1:m, j = 1:n)
 rows[1 : n*m] .= getindex.(I, 1)[:]
 cols[1 : n*m] .= getindex.(I, 2)[:]
 return rows, cols
end

function jprod!(nlp :: RlxMPCC,
	            x   :: AbstractVector,
				v   :: AbstractVector,
				Jv  :: AbstractVector)
  increment!(nlp, :neval_jprod)
  Jv .= jac(nlp, x) * v
  return Jv
end

function jtprod!(nlp :: RlxMPCC,
	             x   :: AbstractVector,
				 v   :: AbstractVector,
				 Jtv :: AbstractVector)
  increment!(nlp, :neval_jtprod)
  Jtv .= jac(nlp,x)' * v
  return Jtv
end

function hess(nlp        :: RlxMPCC,
	          x          :: AbstractVector;
			  obj_weight :: Real = one(eltype(x)))
  increment!(nlp, :neval_hess)
  Hx = hess(nlp.mod, x, obj_weight = obj_weight)
  return tril(Hx)
end

function hess(nlp        :: RlxMPCC,
	          x          :: AbstractVector,
			  y          :: AbstractVector;
			  obj_weight :: Real = one(eltype(x)))
  increment!(nlp, :neval_hess)
  ncc, nlc, nvar = nlp.mod.meta.ncc, nlp.mod.meta.ncon, nlp.mod.meta.nvar

  if ncc > 0
   JGx, JHx = jacG(nlp.mod, x),   jacH(nlp.mod, x)
   Gx,  Hx  = consG(nlp.mod, x), consH(nlp.mod, x)

   lG, lH = y[nlc+1:nlc+ncc], y[nlc+ncc+1:nlc+2*ncc]
   lphi   = y[nlc+2*ncc+1:nlc+3*ncc]

   dgg, dgh, dhh = ddphi(Gx, Hx, nlp.r, nlp.s, nlp.t)
   dp            =  dphi(Gx, Hx, nlp.r, nlp.s, nlp.t)

   ny = vcat(y[1:nlc], lG - lphi .* dp[1:ncc], lH - lphi .* dp[ncc+1:2*ncc])
  else
   ny = y[1:nlc]
  end

  Hx = hess(nlp.mod, x, ny, obj_weight = obj_weight)

  if ncc > 0
   hess_P = JGx' * (diagm(0 => (dgg .* lphi)) * JGx)
          + JHx' * (diagm(0 => (dgh .* lphi)) * JHx)
	 	  + JGx' * (diagm(0 => (dgh .* lphi)) * JHx)
	  	  + JHx' * (diagm(0 => (dgh .* lphi)) * JGx)
   #for i=1:ncc
   # Hx += lphi[i] * (JGx[i,:] * JHx[i,:]' + JHx[i,:] * JGx[i,:]')
   #end
   Hx += hess_P
  end

  return tril(Hx)
end

function hess_structure!(nlp  :: RlxMPCC,
	                     rows :: AbstractVector{<: Integer},
						 cols :: AbstractVector{<: Integer})
  n = nlp.meta.nvar
  I = ((i,j) for i = 1:n, j = 1:n if i ≥ j)
  rows .= getindex.(I, 1)
  cols .= getindex.(I, 2)
  return rows, cols
end

function hess_coord!(nlp        :: RlxMPCC,
	                 x          :: AbstractVector,
					 vals       :: AbstractVector;
					 obj_weight :: Real = one(eltype(x)))
  increment!(nlp, :neval_hess)
  Hx = hess(nlp, x, obj_weight=obj_weight)
  k = 1
  for j = 1 : nlp.meta.nvar
    for i = j : nlp.meta.nvar
      vals[k] = Hx[i, j]
      k += 1
    end
  end
  return vals
end

function hess_coord!(nlp        :: RlxMPCC,
	                 x          :: AbstractVector,
					 y          :: AbstractVector,
					 vals       :: AbstractVector;
					 obj_weight :: Real = one(eltype(x)))
  increment!(nlp, :neval_hess)
  Hx = hess(nlp, x, y, obj_weight=obj_weight)
  k = 1
  for j = 1 : nlp.meta.nvar
    for i = j : nlp.meta.nvar
      vals[k] = Hx[i, j]
      k += 1
    end
  end
  return vals
end

function hprod!(nlp        :: RlxMPCC,
	            x          :: AbstractVector,
				v          :: AbstractVector,
				Hv         :: AbstractVector;
				obj_weight :: Real = one(eltype(x)))
  increment!(nlp, :neval_hprod)
  Hx = hess(nlp, x, obj_weight = obj_weight)
  Hv .= (Hx + Hx' - diagm(0 => diag(Hx))) * v
  return Hv
end

function hprod!(nlp        :: RlxMPCC,
	            x          :: AbstractVector,
				y          :: AbstractVector,
				v          :: AbstractVector,
				Hv         :: AbstractVector;
				obj_weight :: Real = one(eltype(x)))
  increment!(nlp, :neval_hprod)
  Hx = hess(nlp, x, y, obj_weight = obj_weight)
  Hv .= (Hx + Hx' - diagm(0 => diag(Hx))) * v
  return Hv
end

"""
Return the violation of the constraints
lb   <= x    <= ub,
lc   <= c(x) <= uc,
lccG <= G(x) + tb,
lccH <= H(x) + tb,
Phi(G(x),H(x),r s, t) <= 0.
"""
function viol(nlp :: RlxMPCC, x :: AbstractVector)

 mod, n, ncc = nlp.mod, nlp.mod.meta.nvar, nlp.mod.meta.ncc
 x = length(x) == n ? x : x[1:n]

 feas_x = vcat(max.(mod.meta.lvar-x, 0), max.(x-mod.meta.uvar, 0))

 if mod.meta.ncon > 0

  c = cons_nl(mod, x)
  feas_c = vcat(max.(mod.meta.lcon-c, 0), max.(c-mod.meta.ucon, 0))

 else

  feas_c = Float64[]

 end

 if ncc > 0

  Gx, Hx = consG(mod, x), consH(mod, x)

  feas_cp = vcat(max.(mod.meta.lccG - Gx .- nlp.tb, 0),
                 max.(mod.meta.lccH - Hx .- nlp.tb, 0))
  feas_cc = max.(phi(Gx, Hx, nlp.r, nlp.s, nlp.t), 0)
 else
  feas_cp = Float64[]
  feas_cc = Float64[]
 end

 return vcat(feas_x, feas_c, feas_cp, feas_cc)
end
