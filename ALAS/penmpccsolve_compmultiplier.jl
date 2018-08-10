function lsq_computation_multiplier_bool(ps  :: PenMPCCSolve,
                                         xjk :: Vector)

   gradpen = copy(ps.rpen.gx)

   l = _lsq_computation_multiplier(ps, gradpen, xjk)

   l_negative = findfirst(x->x<0, l) != 0

 return l, l_negative
end

function _lsq_computation_multiplier(ps      :: PenMPCCSolve,
                                     gradpen :: Vector,
                                     xj      :: Vector)

 r,s,t = ps.pen.r,ps.pen.s,ps.pen.t
 n, ncc = ps.n, ps.ncc

 nx1 = length(ps.wn1)
 nx2 = length(ps.wn2)
 nx  = nx1 + nx2

 wn1, wn2 = ps.wn1, ps.wn2
 w1, w2   = ps.w1, ps.w2
 wcomp    = ps.wcomp
 nw1, nw2, nwcomp = length(ps.w1), length(ps.w2), length(ps.wcomp)

 #jacobian of active constraints : (lx,ux,lg,lh,lphi)'*J=grad
 dg = dphi(xj[n+1:n+ncc], xj[n+ncc+1:n+2*ncc], r, s, t)
 gx, gy = dg[1:ncc], dg[ncc+1:2*ncc]

 Jxl = -diagm(ones(n))[1:n,wn1]
 Jxu = diagm(ones(n))[1:n,wn2]

 JG  = -diagm(ones(ncc))[1:ncc,w1]
 JPG = diagm(collect(gx))[1:ncc,wcomp]
 JH  = -diagm(ones(ncc))[1:ncc,w2]
 JPH = diagm(collect(gy))[1:ncc,wcomp]

 Jx=hcat(Jxl, zeros(n,nx2+nw1+nw2+2*nwcomp)) + 
    hcat(zeros(n,nx1), Jxu, zeros(n,nw1+nw2+2*nwcomp))

 Tmpsg=hcat(zeros(ncc,nx),JG,zeros(ncc,nw2+2*nwcomp)) +
       hcat(zeros(ncc,nx+nw1+nw2),JPG,zeros(ncc,nwcomp))
 Tmpsh=hcat(zeros(ncc,nx+nw1),JH,zeros(ncc,2*nwcomp))+
       hcat(zeros(ncc,nx+nw1+nw2+nwcomp),JPH)

 J = vcat(Jx,Tmpsg,Tmpsh)

 #compute the multiplier using pseudo-inverse
 #l=J \ gradpen
 l = - pinv(J) * gradpen

 lk = zeros(2*n+3*ncc)

 lk[wn1]             = l[1:nx1]
 lk[n+wn2]           = l[nx1+1:nx]
 lk[2*n+w1]          = l[nx+1:nx+nw1]
 lk[2*n+ncc+w2]      = l[nx+nw1+1:nx+nw1+nw2]
 lk[2*n+2*ncc+wcomp] = l[nx+nw1+nw2+1:nx+nw1+nw2+nwcomp]

 return lk
end
