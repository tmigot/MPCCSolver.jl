#Tests des packages
include("../ALAS/include.jl")

using MPCCmod
using OutputRelaxationmod
#package pour plot
using PyPlot

using NLPModels
using JuMP
using NLPModels
using Ipopt
using NLopt

path_to_data=""
include(string(path_to_data,"Portfolio.jl"))

#Algo choices:
solvers=["Ipopt"]
allrelaxations=["NL","SS","KS","ButterflyTheta1","ButterflyTheta1_C1","ButterflyTheta1_C2","Thetal0","Theta2l0","Thetal0scaled","Theta2l0scaled"];
thetal0=["Thetal0","Theta2l0","Thetal0scaled","Theta2l0scaled"]
relaxations=["ButterflyALAS"]
#problem infeasible with NL

pbs200=["size200/orl200-05-a","size200/orl200-005-a","size200/orl200-05-b","size200/orl200-005-b","size200/orl200-05-c","size200/orl200-005-c","size200/orl200-05-d","size200/orl200-005-d","size200/orl200-05-e","size200/orl200-005-e","size200/orl200-05-f","size200/orl200-005-f","size200/orl200-05-g","size200/orl200-005-g","size200/orl200-05-h","size200/orl200-005-h","size200/orl200-05-i","size200/orl200-005-i","size200/orl200-05-j","size200/orl200-005-j","size200/pard200_a","size200/pard200_b","size200/pard200_c","size200/pard200_d","size200/pard200_e","size200/pard200_f","size200/pard200_g","size200/pard200_h","size200/pard200_i","size200/pard200_j"]
pbs300=["size300/orl300_05_a","size300/orl300_005_a","size300/orl300_05_b","size300/orl300_005_b","size300/orl300_05_c","size300/orl300_005_c","size300/orl300_05_d","size300/orl300_005_d","size300/orl300_05_e","size300/orl300_005_e","size300/orl300_05_f","size300/orl300_005_f","size300/orl300_05_g","size300/orl300_005_g","size300/orl300_05_h","size300/orl300_005_h","size300/orl300_05_i","size300/orl300_005_i","size300/orl300_05_j","size300/orl300_005_j","size300/pard300_a","size300/pard300_b","size300/pard300_c","size300/pard300_d","size300/pard300_e","size300/pard300_f","size300/pard300_g","size300/pard300_h","size300/pard300_i","size300/pard300_j"]
pbs400=["size400/orl400_05_a","size400/orl400_005_a","size400/orl400_05_b","size400/orl400_005_b","size400/orl400_05_c","size400/orl400_005_c","size400/orl400_05_d","size400/orl400_005_d","size400/orl400_05_e","size400/orl400_005_e","size400/orl400_05_f","size400/orl400_005_f","size400/orl400_05_g","size400/orl400_005_g","size400/orl400_05_h","size400/orl400_005_h","size400/orl400_05_i","size400/orl400_005_i","size400/orl400_05_j","size400/orl400_005_j","size400/pard400_a","size400/pard400_b","size400/pard400_c","size400/pard400_d","size400/pard400_e","size400/pard400_f","size400/pard400_g","size400/pard400_h","size400/pard400_i","size400/pard400_j"]
pbs=vcat(pbs200,pbs300,pbs400)

names=["VaR","CVaR","RVaR","RCVar"];
beta=[0.9 0.95 0.99];
cb=[1.2816 1.6449 2.3263;1.7550 2.0627 2.6652;1.3333 2.0647 4.9247;3.0000 4.3589 9.9499];

#sorties
npb=length(pbs);npb=1;
nrelax=length(relaxations);nrelax=1;
nsolv=length(solvers);nsolv=1;
nmod=size(cb,1);nmod=1;
nbeta=length(beta);nbeta=1;

obj=zeros(npb,nrelax,nsolv,nmod);
tm=zeros(npb,nrelax,nsolv,nmod);
feas=zeros(npb,nrelax,nsolv,nmod);

prec=1e-6
r0=0.1
s0=0.1
#t0=sqrt(0.1) #t=o(r) normalement
t0=0.01
sig_r=0.1
sig_s=0.1
sig_t=0.01

for mo=1:nmod
 for b=1:nbeta
 cbmo=cb[mo,b]
  for p=1:npb
	pb=pbs[p]
	@printf("%f %s \n",p,pb)

	#nous les vecteurs l et u
	bds=readdlm(string("data/mv/",pb,".bds"));
	u=bds[:,2];

	# nous donne la matrice Q
	Q=readdlm(string("data/mv/",pb,".mat"))
	n=size(Q,1);

	txt=readdlm(string(path_to_data,"data/mv/",pb,".txt"));
	mu=txt[:,1];

	kappa=20;
#n=5;Q=Q[1:n,1:n];mu=mu[1:n];u=u[1:n];kappa=2;
	#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	#%CHOIX DU POINT INITIAL
	x0=prec*ones(n);y0=ones(n);z0=vcat(x0,y0)
	#%CHOIX DE LA GESTION DES PARAMETRES (r,s,t,tb)
	nlp,G,H=Portfolio(cbmo,mu,Q,u,kappa,z0,prec)

	#CREATION DU MPCCmod:
	omcc=MPCCmod.MPCC(nlp,G,H)

	#APPEL DU SOLVER:
@time xb,fb,orb,nb_eval = MPCCsolve.solve(omcc,r0,sig_r,s0,sig_s,t0,sig_t)

	try
 	oa=orb.inner_output_alas[1]
 	@printf("%s	",orb.solve_message)
	end
	@show nb_eval

  end
 end
end

