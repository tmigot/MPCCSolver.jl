function Newton(nlp :: AbstractNLPModel;
                stp :: TStopping=TStopping(),
                verbose :: Bool=false,
                verboseLS :: Bool = false,
                verboseCG :: Bool = false,
                linesearch :: Function = Newarmijo_wolfe,
                Nwtdirection :: Function = NwtdirectionCG,
                hessian_rep :: Function = hessian_operator,
                kwargs...)

    x = copy(nlp.meta.x0)
    n = nlp.meta.nvar

    # xt = Array(Float64, n)
    # ∇ft = Array(Float64, n)
    xt = Array{Float64}(n)
    ∇ft = Array{Float64}(n)

    f = obj(nlp, x)
    #∇f = grad(nlp, x)
    stp, ∇f = start!(nlp,stp,x)
    ∇fNorm = BLAS.nrm2(n, ∇f, 1)

    H = hessian_rep(nlp,x)

    iter = 0


    verbose && @printf("%4s  %8s  %7s  %8s  %4s %8s\n", " iter", "f", "‖∇f‖", "∇f'd", "bk","t")
    verbose && @printf("%5d  %8.1e  %7.1e", iter, f, ∇fNorm)

    optimal, unbounded, tired, elapsed_time = stop(nlp,stp,iter,x,f,∇f)

    OK = true
    stalled_linesearch = false
    stalled_ascent_dir = false

    β = 0.0
    d = zeros(∇f)
    scale = 1.0

    h = LineModel(nlp, x, d)

    while (OK && !(optimal || tired || unbounded))
        d = Nwtdirection(H,∇f,verbose=verboseCG)
        slope = BLAS.dot(n, d, 1, ∇f, 1)

        verbose && @printf("  %8.1e", slope)

        h = Optimize.redirect!(h, x, d)

        verboseLS && println(" ")

        t, t_original, good_grad, ft, nbk, nbW, stalled_linesearch = linesearch(h, f, slope, ∇ft; verboseLS = verboseLS, kwargs...)

        if linesearch in interfaced_ls_algorithms
          ft = obj(nlp, x + (t)*d)
          nlp.counters.neval_obj += -1
        end

        if verboseLS
           (verbose) && print(" \n")
         else
           (verbose) && @printf("  %4d %8s\n", nbk,t)
         end


        BLAS.blascopy!(n, x, 1, xt, 1)
        BLAS.axpy!(n, t, d, 1, xt, 1)
        good_grad || (∇ft = grad!(nlp, xt, ∇ft))

        # Move on.
        s = xt - x
        y = ∇ft - ∇f
        β = (∇ft⋅y) / (∇f⋅∇f)
        x = xt
        f = ft

        H = hessian_rep(nlp,x)

        BLAS.blascopy!(n, ∇ft, 1, ∇f, 1)

        # norm(∇f) bug: https://github.com/JuliaLang/julia/issues/11788
        ∇fNorm = BLAS.nrm2(n, ∇f, 1)
        iter = iter + 1

        verbose && @printf("%4d  %8.1e  %7.1e", iter, f, ∇fNorm)
        optimal, unbounded, tired, elapsed_time = stop(nlp,stp,iter,x,f,∇f)
        OK = !stalled_linesearch & !stalled_ascent_dir

    end

    verbose && @printf("\n")

    if optimal status = :Optimal
    elseif unbounded status = :Unbounded
    elseif stalled_linesearch status = :StalledLinesearch
    elseif stalled_ascent_dir status = :StalledAscentDir
    else status = :UserLimit
    end

    return (x, f, stp.optimality_residual(∇f), iter, optimal, tired, status, h.counters.neval_obj, h.counters.neval_grad, h.counters.neval_hess)
end

function Newarmijo_wolfe(h :: LineModel,
                         h₀ :: Float64,
                         slope :: Float64,
                         g :: Array{Float64,1};
                         τ₀ :: Float64=1.0e-4,
                         τ₁ :: Float64=0.9999,
                         bk_max :: Int=50,
                         nbWM :: Int=50,
                         verboseLS :: Bool=false,
                         check_slope :: Bool = false,
                         kwargs...)

    if check_slope
      (abs(slope - grad(h, 0.0)) < 1e-4) || warn("wrong slope")
      verboseLS && @show h₀ obj(h, 0.0) slope grad(h,0.0)
    end

    # Perform improved Armijo linesearch.
    nbk = 0
    nbW = 0
    t = 1.0

    # First try to increase t to satisfy loose Wolfe condition
    ht = obj(h, t)
    slope_t = grad!(h, t, g)
    while (slope_t < τ₁*slope) && (ht <= h₀ + τ₀ * t * slope) && (nbW < nbWM)
        t *= 5.0
        ht = obj(h, t)
        slope_t = grad!(h, t, g)

        nbW += 1
        verboseLS && @printf(" W  %4d  slope  %4d slopet %4d\n", nbW, slope, slope_t);
    end

    hgoal = h₀ + slope * t * τ₀;
    fact = -0.8
    ϵ = 1e-10

    # Enrich Armijo's condition with Hager & Zhang numerical trick
    Armijo = (ht <= hgoal) || ((ht <= h₀ + ϵ * abs(h₀)) && (slope_t <= fact * slope))
    good_grad = true
    while !Armijo && (nbk < bk_max)
        t *= 0.4
        ht = obj(h, t)
        hgoal = h₀ + slope * t * τ₀;

        # avoids unused grad! calls
        Armijo = false
        good_grad = false
        if ht <= hgoal
            Armijo = true
        elseif ht <= h₀ + ϵ * abs(h₀)
            slope_t = grad!(h, t, g)
            good_grad = true
            if slope_t <= fact * slope
                Armijo = true
            end
        end

        nbk += 1
        verboseLS && @printf(" A  %4d  h0  %4e ht %4e\n", nbk, h₀, ht);
    end

    verboseLS && @printf("  %4d %4d %8e\n", nbk, nbW, t);
    stalled = (nbk == bk_max)
    @assert (t > 0.0) && (!isnan(t)) "invalid step"
    return (t, t, good_grad, ht, nbk, nbW, stalled)#,h.f_eval,h.g_eval,h.h_eval)
end

function NwtdirectionCG(H,∇f;verbose::Bool=false)
    e=1e-6
    n = length(∇f)
    τ = 0.5 # need parametrization
    cgtol = max(e, min(0.7, 0.01 * norm(∇f)^(1.0 + τ)))
    
    (d, cg_stats) = cgTN(H, -∇f,
                       atol=cgtol, rtol=0.0,
                       itmax=max(2 * n, 50),
                       verbose=verbose)
    
    return d
end
