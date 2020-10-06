"""
fillMstat: fill-in the MPCCAtX state structure for the M-stationary conditions.

`fillMstat(:: AbstractMPCCModel, :: MPCCAtX; pnorm :: Float64 = Inf)`

"""
function fillMstat(pb    :: AbstractMPCCModel,  #:: AbstractNLPModel,
                   state :: MPCCAtX;
                   pnorm :: Float64 = Inf)

    ncon, ncc = pb.meta.ncon, pb.meta.ncc

    update!(state, gx = grad(pb, state.x)) #utiliser la fonction fill_in ?

    if length(state.lambda)+length(state.mu)+length(state.lG)+length(state.lH) == 0
     return norm(state.gx, pnorm)
    else
     n = length(state.x)
     pr = 1e-3

     wl = findall(x->(norm(x) <= pr), abs.(-state.x + pb.meta.lvar))
     wu = findall(x->(norm(x) <= pr), abs.(-state.x + pb.meta.uvar))
     lui = zeros(n); lui[wl] .= -1.0; lui[wu] .= 1.0; LUi = (diagm(0 => lui))
     #LUi = (diagm(0 => lui))[:,union(wl,wu)]

     if ncon > 0
      update!(state, Jx = jac(pb, state.x), cx = cons_nl(pb, state.x))
      Jx = zeros(size(state.Jx'))
      wc1 = findall(x->norm(x) <= pr, abs.(state.cx-pb.meta.ucon))
      wc2 = findall(x->norm(x) <= pr, abs.(state.cx-pb.meta.lcon))
      Jx[:,wc1] = (state.Jx')[:,wc1]
      Jx[:,wc2] = -(state.Jx')[:,wc2]
     end

     if ncc > 0
      update!(state, cGx = consG(pb, state.x), cHx = consH(pb, state.x))
      Jx = ncon > 0 ? Jx : zeros(size(state.Jx'))
      wG = findall(x->norm(x) <= pr, abs.(state.Gc))
      wH = findall(x->norm(x) <= pr, abs.(state.Hc))
      Jx[:,ncon.+wG] = (state.Jx')[:,ncon.+wG]
      Jx[:,ncon.+ncc.+wH] = (state.Jx')[:,ncon.+ncc.+wH]
     end

    if ncon + ncc >0
      Jct = hcat(LUi,Jx)
    else
      Jct = LUi
    end


     l = pinv(Jct) * (- state.gx)
     gLagx1 = Jct*l+state.gx
     res_nonlin1 = min.(vcat(l[n+wc1],l[n+wc2]),0.0)

     if !(true in isnan.(vcat(state.mu,state.lambda,state.lG,state.lH)))
      gLagx2 = Jct*vcat(state.mu, state.lambda,state.lG,state.lH)+state.gx
      res_nonlin2 = min.(vcat(state.lambda[wc1],state.lambda[wc2],state.lG[wG],state.lH[wH]),0.0)
     else
      gLagx2 = Inf
      res_nonlin2 = Inf
     end

     if norm(res_nonlin1) <= norm(res_nonlin2)
      state.mu, state.lambda = l[1:n], l[n+1:n+ncon]
      state.lG, state.lH     = l[n+ncon+1:n+ncon+ncc], l[n+ncon+ncc+1:n+ncon+2*ncc]
      gLagx = gLagx1
      res_nonlin = res_nonlin1
     else
      gLagx = gLagx2
      res_nonlin = res_nonlin2
     end
    end

    res_bounds = min.(state.mu,0.0)

    if ncon > 0
     Ib = ∩(wG,wH)
     res_mpcc = min.(state.lG[Ib].*state.lH[Ib],min.(state.lG[Ib],0)+min.(state.lH[Ib],0))
     feas = vcat(max.(state.cx   - pb.meta.ucon,0),max.(- state.cx + pb.meta.lcon,0),max.(  state.x - pb.meta.uvar,0), max.(- state.x + pb.meta.lvar,0))
    else
     res_mpcc = 0.0
     feas = vcat(max.(  state.x - pb.meta.uvar,0), max.(- state.x + pb.meta.lvar,0))
    end

    if ncc > 0
     res_mpcc = 0.0
     feas_mpcc = vcat(max.(-state.Gc,0),max.(-state.Gc,0),max.(state.Gc.*state.Hc,0))
    else
     res_mpcc = 0.0
     feas_mpcc = 0.0
    end
@show gLagx, feas, feas_mpcc, res_bounds, res_nonlin, res_mpcc
    res = vcat(gLagx, feas, feas_mpcc, res_bounds, res_nonlin, res_mpcc)
    if norm(res_nonlin) > pr
     #Z = nullspace(Jct)
     #I need to take care of this...
    end
@show norm(res, pnorm)
    return norm(res, pnorm)
end
