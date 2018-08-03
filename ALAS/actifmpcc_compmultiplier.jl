function lsq_computation_multiplier_bool(ma  :: ActifMPCC,
                                         xjk :: Vector)

   gradpen = ma.rpen.gx

   l = _lsq_computation_multiplier(ma, gradpen, xjk)

   l_negative = findfirst(x->x<0, l) != 0

 return l, l_negative
end

function _lsq_computation_multiplier(ma      :: ActifMPCC,
                                     gradpen :: Vector,
                                     xj      :: Vector)

 r,s,t = ma.pen.r,ma.pen.s,ma.pen.t

 dg = dphi(xj[ma.n+1:ma.n+ma.ncc],
           xj[ma.n+ma.ncc+1:ma.n+2*ma.ncc],
           r, s, t)

 gx = dg[1:ma.ncc]
 gy = dg[ma.ncc+1:2*ma.ncc]

 #matrices des contraintes actives : (lx,ux,lg,lh,lphi)'*A=b
 nx1 = length(ma.wn1)
 nx2 = length(ma.wn2)
 nx  = nx1 + nx2

 nw1 = length(ma.w1)
 nw2 = length(ma.w2)
 nwcomp = length(ma.wcomp)

 Dlg = -diagm(ones(nw1))
 Dlh = -diagm(ones(nw2))
 Dlphig = diagm(collect(gx)[ma.wcomp])
 Dlphih = diagm(collect(gy)[ma.wcomp])

 #matrix of size: nc+nw1+nw2+nwcomp x nc+nw1+nw2+2nwcomp
 A=[hcat(-diagm(ones(nx1)),zeros(nx1,nx2+nw1+nw2+2*nwcomp));
    hcat(zeros(nx2,nx1),diagm(ones(nx2)),zeros(nx2,nw1+nw2+2*nwcomp));
    hcat(zeros(nw1,nx),Dlg,zeros(nw1,nw2+2*nwcomp));
    hcat(zeros(nw2,nx),zeros(nw2,nw1),Dlh,zeros(nw2,2*nwcomp));
    hcat(zeros(nwcomp,nx),zeros(nwcomp,nw1+nw2),Dlphig,Dlphih)]

 #vector of size: nc+nw1+nw2+2nwcomp
 b=-[gradpen[ma.wn1];
     gradpen[ma.wn2];
     gradpen[ma.w1+ma.n];
     gradpen[ma.wcomp+ma.n];
     gradpen[ma.w2+ma.n+ma.ncc];
     gradpen[ma.wcomp+ma.n+ma.ncc]] 

 #compute the multiplier using pseudo-inverse
 l=pinv(A')*b
 #l=A' \ b

 lk = zeros(2*ma.n+3*ma.ncc)
 lk[ma.wn1] = l[1:nx1]
 lk[ma.n+ma.wn2] = l[nx1+1:nx]
 lk[2*ma.n+ma.w1] = l[nx+1:nx+nw1]
 lk[2*ma.n+ma.ncc+ma.w2] = l[nx+nw1+1:nx+nw1+nw2]
 lk[2*ma.n+2*ma.ncc+ma.wcomp] = l[nx+nw1+nw2+1:nx+nw1+nw2+nwcomp]

 return lk
end
