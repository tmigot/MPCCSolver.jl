export Portfolio

function Portfolio(cbmo::Float64,mu::Vector,
                   Q::Any,u::Vector{Float64},
                   kappa::Int64,z0::Vector{Float64}
                   ;relax::Function=NL,
                   t::Float64=0.0,
		   n::Int64=5,
		   tol::Float64=1e-6)

	c1=z->sum(z[1:n]);lc1=1.0;uc1=1.0;
	c2=z->sum(z[n+1:2*n]);lc2=n-kappa;uc2=Inf;
	c3=z->z[1:n].*z[n+1:2*n];lc3=-Inf*ones(n);uc3=zeros(n);
	#c3=z->sum(z[1:n].*z[n+1:2*n]);lc3=-Inf;uc3=zeros(1);
	c3=z->relax(z[1:n],z[n+1:2*n],t=t);lc3=-Inf*ones(n);uc3=zeros(n);
	cons=z->vcat(c1(z),c2(z),c3(z));
	fobj=z->cbmo*sqrt(z[1:n]'*Q*z[1:n])-z[1:n]'*mu;
	lvar=zeros(2*n);uvar=vcat(u,ones(n));
	mp=ADNLPModel(fobj, z0, lvar=lvar, uvar=uvar, c=cons, lcon=vcat(lc1,lc2,lc3), ucon=vcat(uc1,uc2,uc3))

 return mp
end

function Portfolio(cbmo::Float64,mu::Vector,
                   Q::Any,u::Vector{Float64},
                   kappa::Int64,z0::Vector{Float64},
		   tol::Float64=1e-6)
	n=size(Q,1)

	c1=z->sum(z[1:n]);lc1=1.0;uc1=1.0;
	c2=z->sum(z[n+1:2*n]);lc2=n-kappa;uc2=Inf;
	c3=z->z[1:n].*z[n+1:2*n];lc3=-Inf*ones(n);uc3=zeros(n);
	cons=z->vcat(c1(z),c2(z),c3(z));
	fobj=z->cbmo*sqrt(z[1:n]'*Q*z[1:n])-z[1:n]'*mu;
	lvar=zeros(2*n);uvar=vcat(u,ones(n));
	mp=ADNLPModel(fobj, z0, lvar=lvar, uvar=uvar, c=cons, lcon=vcat(lc1,lc2,lc3), ucon=vcat(uc1,uc2,uc3))

	#ex1=JuMP.Model()
	#JuMP.@variable(ex1,x[1:2],start=1.0)
	#JuMP.@NLobjective(ex1,Min,x[1]-x[2])
	#JuMP.@constraint(ex1,1-x[2]>=0)
	#ex1=MathProgNLPModel(ex1)

	#cG=z->z[1:n]
	#G=ADNLPModel(x->(), z0, c=cG, lcon=zeros(n))
	#cH=z->z[n+1:2*n]
	#H=ADNLPModel(x->(), z0, c=cH, lcon=zeros(n))

	G=JuMP.Model()
	JuMP.@variable(G,x[1:2*n],start=1.0)
	JuMP.@constraint(G,x[1:n].>=0)
	JuMP.@NLobjective(G,Min,0.0)
	G=MathProgNLPModel(G)
	H=JuMP.Model()
	JuMP.@variable(H,x[1:2*n],start=1.0)
	JuMP.@constraint(H,x[n+1:2*n].>=0)
	JuMP.@NLobjective(H,Min,0.0)
	H=MathProgNLPModel(H)

 return mp,G,H
end

