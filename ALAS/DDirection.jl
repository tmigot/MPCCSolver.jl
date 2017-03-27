"""
Package de fonctions pour calculer des directions de descentes :
les fonctions doivent contenir deux duplicats : le 1er calcul la direction, la 2ème le beta.

SteepestDescent(ma::ActifMPCCmod.MPCC_actif,g::Vector,xj::Vector,hd::Any,beta::Float64)
SteepestDescent(beta,gradft,gradf,y,d) 

CGFR(ma::ActifMPCCmod.MPCC_actif,g::Vector,xj::Vector,hd::Any,beta::Float64)
CGFR(beta,gradft,gradf,y,d) 

CGPR(ma::ActifMPCCmod.MPCC_actif,g::Vector,xj::Vector,hd::Any,beta::Float64)
CGPR(beta,gradft,gradf,y,d) 

CGHS(ma::ActifMPCCmod.MPCC_actif,g::Vector,xj::Vector,hd::Any,beta::Float64)
CGHS(beta,gradft,gradf,y,d) 

CGHZ(ma::ActifMPCCmod.MPCC_actif,g::Vector,xj::Vector,hd::Any,beta::Float64)
CGHZ(beta,gradft,gradf,y,d) 

NwtdirectionSpectral(ma::ActifMPCCmod.MPCC_actif,g::Vector,xj::Vector,hd::Any,beta::Float64)
NwtdirectionSpectral(H,g,hd,beta)

NwtdirectionMA57(ma::ActifMPCCmod.MPCC_actif,g::Vector,xj::Vector,hd::Any,beta::Float64)  #problème définition MA57 matrice ?
NwtdirectionMA57(H,g,hd,beta)

NwtdirectionLDLt(ma::ActifMPCCmod.MPCC_actif,g::Vector,xj::Vector,hd::Any,beta::Float64)
NwtdirectionLDLt(H,g,hd,beta)

Fonctions additionnelles :
ldlt_symm(A0 :: Array{Float64,2}, piv :: Char='r')
"""
module DDirection

using ActifMPCCmod

"""
SteepestDescent : Calcul une direction de descente
"""
function SteepestDescent(ma::ActifMPCCmod.MPCC_actif,g::Vector,xj::Vector,hd::Any,beta::Float64)
 return -g
end

function SteepestDescent(beta,gradft,gradf,y,d,step::Float64) 
 return 0.0
end

"""
Gradient Conjugué : formule Fletcher and Reeves
"""
function CGFR(ma::ActifMPCCmod.MPCC_actif,g::Vector,xj::Vector,hd::Any,beta::Float64)
 return - g + beta*hd
end

function CGFR(ma::ActifMPCCmod.MPCC_actif,beta::Float64,gradft,gradf,y,d,step::Float64) 

 if dot(gradft,gradf)<0.2*dot(gradft,gradft) # Powell restart
  #Formula FR
  #β = (∇ft⋅∇ft)/(∇f⋅∇f) FR
  beta=dot(gradft,gradft)/dot(gradf,gradf)
 end

 return ActifMPCCmod.setbeta(ma,beta)
end

"""
Gradient Conjugué : Calcul une direction de descente
"""
function CGPR(ma::ActifMPCCmod.MPCC_actif,g::Vector,xj::Vector,hd::Any,beta::Float64)
 return - g + beta*hd
end

function CGPR(ma::ActifMPCCmod.MPCC_actif,beta::Float64,gradft,gradf,y,d,step::Float64) 

 if dot(gradft,gradf)<0.2*dot(gradft,gradft) # Powell restart
  #Formula PR
  #β = (∇ft⋅y)/(∇f⋅∇f)
  beta=dot(gradft,y)/dot(gradf,gradf)
 end

 return ActifMPCCmod.setbeta(ma,beta)
end

"""
Gradient Conjugué : Calcul une direction de descente
"""
function CGHS(ma::ActifMPCCmod.MPCC_actif,g::Vector,xj::Vector,hd::Any,beta::Float64)
 return - g + beta*hd
end

function CGHS(ma::ActifMPCCmod.MPCC_actif,beta::Float64,gradft,gradf,y,d,step::Float64)  

 if dot(gradft,gradf)<0.2*dot(gradft,gradft) # Powell restart
  #Formula HS
  #β = (∇ft⋅y)/(d⋅y)
  beta=dot(gradft,y)/dot(d,y)
 end

 return ActifMPCCmod.setbeta(ma,beta)
end

"""
Gradient Conjugué : Calcul une direction de descente
"""
function CGHZ(ma::ActifMPCCmod.MPCC_actif,g::Vector,xj::Vector,hd::Any,beta::Float64)
 return - g + beta*hd
end

function CGHZ(ma::ActifMPCCmod.MPCC_actif,beta::Float64,gradft,gradf,y,d,step::Float64) 

 if dot(gradft,gradf)<0.2*dot(gradft,gradft) # Powell restart
  #Formula HZ
  n2y = dot(y,y)
  b1 = dot(y,d)
  #β = ((y-2*d*n2y/β1)⋅∇ft)/β1
  beta = dot(y-2*d*n2y/b1,gradft)/b1
 end

 return ActifMPCCmod.setbeta(ma,beta)
end

"""
Direction de quasi-Newton:
y=gradft-gradf

EN TRAVAUX
"""
function quasiNwt(ma::ActifMPCCmod.MPCC_actif,beta::Float64,gradft,gradf,y,d,step::Float64) 
 qk=y
 pk=step*d #dépend de la direction qui n'est pas de la bonne taille !
 H=ma.Hess
 return ActifMPCCmod.sethess(ma,H)
end

function quasiNwt(ma::ActifMPCCmod.MPCC_actif,g::Vector,xj::Vector,hd::Any,beta::Float64)
    H=ma.Hess
    
    d = - H\g
    return d
end

"""
Direction de Newton Spectral:
"""
function NwtdirectionSpectral(ma::ActifMPCCmod.MPCC_actif,beta::Float64,gradft,gradf,y,d,step::Float64)  
 return ma
end

function NwtdirectionSpectral(ma::ActifMPCCmod.MPCC_actif,g::Vector,xj::Vector,hd::Any,beta::Float64)
    H=ActifMPCCmod.hess(ma,xj)

    Δ = ones(g)
    V = ones(H)

    try
        Δ, V = eig(H)
    catch
        Δ, V = eig(H + eye(H))
    end
    ϵ2 =  1.0e-8 
    Γ = 1.0 ./ max(abs(Δ),ϵ2)
    
    d = - (V * diagm(Γ) * V') * (g)
    return d
end

"""
Direction de Newton MA57:
"""
function NwtdirectionMA57(ma::ActifMPCCmod.MPCC_actif,beta::Float64,gradft,gradf,y,d,step::Float64)  
 return ma
end

function NwtdirectionMA57(ma::ActifMPCCmod.MPCC_actif,g::Vector,xj::Vector,hd::Any,beta::Float64)
    H=ActifMPCCmod.hess(ma,xj)

    M = Ma57
    L = SparseMatrixCSC{Float64,Int32}
    D57 = SparseMatrixCSC{Float64,Int32}
    pp = Array(Int32,1)
    s = Array{Float64}
    ρ = Float64
    ncomp = Int64
    
    #H57 = convert(SparseMatrixCSC{Cdouble,Int32}, H)  #  Hard coded Cdouble
    try
        M = Ma57(H,print_level=-1)
        ma57_factorize(M)
    catch
 	println("*******   Problem in MA57_0")
        res = PDataMA57_0()
        res.OK = false
        return res
    end

    try
        (L, D57, s, pp) = ma57_get_factors(M)
    catch
        println("*******   Problem after MA57_0")
        println(" Cond(H) = $(cond(full(H)))")
        res = PDataMA57_0()
        res.OK = false
        return res
    end

    #################  Future object BlockDiag operator?
    vD1 = diag(D57)       # create internal representation for block diagonal D
    vD2 = diag(D57,1)     #

    vQ1 = ones(vD1)       # vector representation of orthogonal matrix Q
    vQ2 = zeros(vD2)      #
    vQ2 = zeros(vD2)      #
    vQ2m = zeros(vD2)     #
    veig = copy(vD1)      # vector of eigenvalues of D, initialized to diagonal of D
                          # if D diagonal, nothing more will be computed
    
    i=1;
    while i<length(vD1)
        if vD2[i] == 0.0
            i += 1
        else
            mA = [vD1[i] vD2[i];vD2[i] vD1[i+1]]  #  2X2 submatrix
            DiagmA, Qma = eig(mA)                 #  spectral decomposition of mA
            veig[i] = DiagmA[1]
            vQ1[i] = Qma[1,1]
            vQ2[i] = Qma[1,2]
            vQ2m[i] = Qma[2,1]
            vQ1[i+1] = Qma[2,2]
            veig[i+1] = DiagmA[2]
            i += 2
        end  
    end

    Q = spdiagm((vQ1,vQ2m,vQ2),[0,-1,1])           # sparse representation of Q
    
    Δ = veig

    ϵ2 =  1.0e-8
    Γ = max(abs(Δ),ϵ2)

    # Ad = P'*L*Q*Δ*Q'*L'*Pd =    -g
    # replace Δ by Γ to ensure positive definiteness
    sg = s .* g

    d̃ = L\sg[pp]
    d̂ = L'\ (Q*(Q'*d̃ ./ Γ))
    d = - d̂[invperm(pp)] .* s

    return d
end

"""
Direction de Newton 1:
"""
function NwtdirectionLDLt(ma::ActifMPCCmod.MPCC_actif,beta::Float64,gradft,gradf,y,d,step::Float64)  
 return ma
end

function NwtdirectionLDLt(ma::ActifMPCCmod.MPCC_actif,g::Vector,xj::Vector,hd::Any,beta::Float64)
    H=ActifMPCCmod.hess(ma,xj)

    L = Array(Float64,2)
    D = Array(Float64,2)
    pp = Array(Int,1)
    ρ = Float64
    ncomp = Int64

    try
        (L, D, pp, rho, ncomp) = ldlt_symm(H,'r')
    catch
 	println("*******   Problem in LDLt")
        #res = PDataLDLt()
        #res.OK = false
        #return res
        return (NaN, NaN, NaN, Inf, false, true, :fail)
    end

    # A[pp,pp] = P*A*P' =  L*D*L'

    if true in isnan(D) 
 	println("*******   Problem in D from LDLt: NaN")
        println(" cond (H) = $(cond(H))")
        #res = PDataLDLt()
        #res.OK = false
        #return res
        return (NaN, NaN, NaN, Inf, false, true, :fail)
    end

    Δ, Q = eig(D)

    ϵ2 =  1.0e-8
    Γ = max(abs(Δ),ϵ2)

    # Ad = P'*L*Q*Δ*Q'*L'*Pd =    -g
    # replace Δ by Γ to ensure positive definiteness
    d̃ = L\g[pp]
    d̂ = L'\ (Q*(Q'*d̃ ./ Γ))
    d = - d̂[invperm(pp)]

    return d
end

"""
Higams' ldlt_symm translated from matlab. Performs  a so called
 BKK  Bounded Bunch Kaufman factorization of A0, that means 
 ||L|| is bounded and bounded away from zero.
"""
function  ldlt_symm(A0 :: Array{Float64,2}, piv :: Char='r')

    #LDLT_SYMM  Block LDL^T factorization for a symmetric indefinite matrix.
    #     Given a Hermitian matrix A,
    #     [L, D, P, RHO, NCOMP] = LDLT_SYMM(A, PIV) computes a permutation P,
    #     a unit lower triangular L, and a real block diagonal D
    #     with 1x1 and 2x2 diagonal blocks, such that  P*A*P' = L*D*L'.
    #     PIV controls the pivoting strategy:
    #       PIV = 'p': partial pivoting (Bunch and Kaufman),
    #       PIV = 'r': rook pivoting (Ashcraft, Grimes and Lewis).
    #     The default is partial pivoting.
    #     RHO is the growth factor.
    #     For PIV = 'r' only, NCOMP is the total number of comparisons.
    
    #     References:
    #     J. R. Bunch and L. Kaufman, Some stable methods for calculating
    #        inertia and solving symmetric linear systems, Math. Comp.,
    #        31(137):163-179, 1977.
    #     C. Ashcraft, R. G. Grimes and J. G. Lewis, Accurate symmetric
    #        indefinite linear equation solvers. SIAM J. Matrix Anal. Appl.,
    #        20(2):513-561, 1998.
    #     N. J. Higham, Accuracy and Stability of Numerical Algorithms,
    #        Second edition, Society for Industrial and Applied Mathematics,
    #        Philadelphia, PA, 2002; chap. 11.
    
    #    This routine does not exploit symmetry and is not designed to be
    #     efficient.
    #
    #    Adapted from N. Higham Matlab code.
    #    JPD april 23 2015


    # Since array contents are mutable and modified,
    # copy the input matrix to keep it unchanged outside the
    # scope of the function

    A = copy(A0)

    # minimal checks for conforming inputs
    isequal(triu(A)',tril(A)) || error("Must supply Hermitian matrix.")
    piv in ['p','r'] || error("Pivoting must be "'p'" or "'r'".")

    n, = size(A)

    k = 1
    D = eye(n,n); 
    L = eye(n,n);  
    if n == 1   D = A; end
    
    pp = collect(1:n)
    
    maxA = norm(A, Inf)
    ρ = maxA;
    
    ncomp = 0;
    s=1
    
    α = (1 + sqrt(17))/8
    while k < n
        (λ, vr) = findmax( abs(A[k+1:n,k]) );
        r = vr[1] + k;
        if λ > 0
            swap = false;
            if abs(A[k,k]) >= α*λ
                s = 1;
            else 
                if piv == 'p'
                    temp = A[k:n,r]; temp[r-k+1] = 0
                    σ = norm(temp, Inf)
                    if α*λ^2 <= abs(A[k,k])*σ
                        s = 1;
                    elseif abs(A[r,r]) >= α*σ
                        swap = true;
                        m1 = k; m2 = r;
                        s = 1;
                    else
                        swap = true;
                        m1 = k+1; m2 = r;
                        s = 2;
                    end
                    if swap
                        A[[m1,m2],:] = A[[m2,m1],:]
                        L[[m1,m2],:] = L[[m2,m1],:]
                        A[:,[m1,m2]] = A[:,[m2,m1]]
                        L[:,[m1,m2]] = L[:,[m2,m1]]
                        
                        pp[[m1,m2]] = pp[[m2,m1]]
                    end
                elseif piv == 'r'
                    j = k;
                    pivot = false;
                    λ_j = λ;
                    while ~pivot
                        (temp1,vr) = findmax( abs(A[k:n,j]) );
                        ncomp = ncomp + n-k;
                        r = vr[1] + k - 1;
                        temp = A[k:n,r]; temp[r-k+1] = 0.0;
                        λᵣ = max(maximum(temp), -minimum(temp))
                        ncomp = ncomp + n-k;
                        if α*λᵣ <= abs(A[r,r])
                            pivot = true;
                            s = 1;
                            A[k,:], A[r,:] = A[r,:], A[k,:]
                            L[k,:], L[r,:] = L[r,:], L[k,:]
                            A[:,k], A[:,r] = A[:,r], A[:,k]
                            L[:,k], L[:,r] = L[:,r], L[:,k]
                            pp[k], pp[r]  = pp[r], pp[k]
                        elseif λ_j == λᵣ
                            pivot = true;
                            s = 2;
                            A[k,:], A[j,:] = A[j,:], A[k,:]
                            L[k,:], L[j,:] = L[j,:], L[k,:]
                            A[:,k], A[:,j] = A[:,j], A[:,k]
                            L[:,k], L[:,j] = L[:,j], L[:,k]
                            pp[k], pp[j]  = pp[j], pp[k]
                            k1 = k+1;
                            A[k1,:], A[r,:] = A[r,:], A[k1,:]
                            L[k1,:], L[r,:] = L[r,:], L[k1,:]
                            A[:,k1], A[:,r] = A[:,r], A[:,k1]
                            L[:,k1], L[:,r] = L[:,r], L[:,k1]
                            pp[k1], pp[r]  = pp[r], pp[k1]
                        else
                            j = r;
                            s = 1
                            λ_j = λᵣ;
                        end
                    end
                end
            end
            
            if s == 1
                
                D[k,k] = A[k,k];
                A[k+1:n,k] = A[k+1:n,k]/A[k,k];
                L[k+1:n,k] = A[k+1:n,k];
                i = k+1:n;
                A[i,i] = A[i,i] - A[i,k:k] * A[k:k,i];
                A[i,i] = 0.5 * (A[i,i] + A[i,i]');
                
            elseif s == 2
                
                E = A[k:k+1,k:k+1];
                D[k:k+1,k:k+1] = E;
                i= k+2:n;
                C = A[i,k:k+1];
                temp = C/E;
                L[i,k:k+1] = temp;
                A[i,k+2:n] = A[i,k+2:n] - temp*C';
                A[i,i] = 0.5 * (A[i,i] + A[i,i]');                         
            end
            
            # Ensure diagonal real (see LINPACK User's Guide, p. 5.17).
            for i=k+s:n
                A[i,i] = real(A[i,i]);
            end
            
            if  k+s <= n
                val, = findmax(abs(A[k+s:n,k+s:n]))
                ρ = max(ρ, val );
            end
            
        else  # Nothing to do.           
            s = 1;
            D[k,k] = A[k,k];            
        end
        
        k = k + s;
        
        if k == n
            D[n,n] = A[n,n];
            break
        end
        
    end
    #P=eye(n)
    #P = P[pp,:]
    return L, D, pp, ρ, ncomp
end

#end of module
end
