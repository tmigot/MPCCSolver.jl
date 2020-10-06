using ActifMPCCmod

f(x)=x[1]-x[2]
r=1.0;s=1.0;t=1.0;
ActifMPCCmod_success=true
println("On teste le module ActifMPCCmod")


nlp=ADNLPModel(f,ones(2), lvar=-Inf*ones(2), uvar=Inf*ones(2), c=c,y0=ones(1),lcon=lcon,ucon=Inf*ones(1))
mpcc = MPCCmod.MPCC(nlp,G,H,1)

##############################################################
#Test constructeur MPCC_actif :
nlp_ma=ADNLPModel(x->mpcc.mp.f(x), [mpcc.mp.meta.x0;-r;1],lvar=[mpcc.mp.meta.lvar;-r*ones(2*mpcc.nb_comp)], uvar=[mpcc.mp.meta.uvar;Inf*ones(2*mpcc.nb_comp)])
ma=ActifMPCCmod.MPCC_actif(nlp_ma,r,s,t,1)
##############################################################
#Test setw et updatew

w_save=sparse(zeros(2,2))
w_save[1,1]=0.0
w_save[1,2]=1.0
ActifMPCCmod.setw(ma,w_save)
ActifMPCCmod.updatew(ma)

w_save2=sparse(zeros(2,2))
w_save2[2,1]=1.0
w_save2[1,2]=0.0
ma.w=w_save2
ActifMPCCmod.updatew(ma)

##############################################################
# Exemple sans contrainte active :
#println("Exemple 1")
xk=[0.8;0.6;0.5;0.45]
dxk=[0.0;0.0;-1.0;0.0]
ActifMPCCmod.setw(ma,sparse(zeros(2,2)))

 #[alpha,w_save]=ActifMPCCmod.PasMax(ma,xk,dxk)
if !(ma.bar_w==[1;2] && ActifMPCCmod.evalx(ma,xk)==xk && norm(ActifMPCCmod.obj(ma,xk)-0.2)<precision && ActifMPCCmod.grad(ma,xk)==[1;-1;0;0] && norm(ActifMPCCmod.PasMax(ma,xk,dxk)[1]-0.3525,Inf)<precision)
 println("Exemple 1")
 println(ma.bar_w) #[1,2]
 println(ActifMPCCmod.evalx(ma,xk)) #doit rentre la même chose que xk
 println(ActifMPCCmod.obj(ma,xk)) #doit donner xk[1]-xk[2]
 println(ActifMPCCmod.grad(ma,xk)) #doit donner [1,-1,0,0]
 println(ActifMPCCmod.PasMax(ma,xk,dxk))

 ActifMPCCmod_success=false
end

#println("Exemple 2")
nlp_ma2=ADNLPModel(x->mpcc.mp.f(x)+x[3]^2+x[4]^2, [1;0;-r;1],lvar=[mpcc.mp.meta.lvar;-r*ones(2*mpcc.nb_comp)], uvar=[mpcc.mp.meta.uvar;Inf*ones(2*mpcc.nb_comp)])
ma2=ActifMPCCmod.MPCC_actif(nlp_ma2,r,s,t,1)

if !(ma2.w13c==[] && ma2.w24c==[1] && ActifMPCCmod.evalx(ma2,[1;0;1])==[1;0;-r;1] && ActifMPCCmod.obj(ma2,[1;0;1])==3 && ActifMPCCmod.grad(ma2,[1;0;1])==[1;-1;2] && norm(ActifMPCCmod.PasMax(ma2,[1;0;1],[0.0;0.0;-1.0])[1]-1.0,Inf)<precision)
 println("Exemple 2")
 println(ma2.w13c) # []
 println(ma2.w24c) # [1]
 println(ActifMPCCmod.evalx(ma2,[1;0;1])) #doit rentre la même chose que xkp
 println(ActifMPCCmod.obj(ma2,[1;0;1])) #doit donner xkp[1]-xkp[2]+xkp[3]^2+xkp[4]^2=3
 #gradient : [1;-1;2x3;2x4]=[1;-1;-2;2]
 println(ActifMPCCmod.grad(ma2,[1;0;1])) #doit donner [1,-1,2]
 println(ActifMPCCmod.PasMax(ma2,[1;0;1],[0.0;0.0;-1.0]))

 ActifMPCCmod_success=false
end

#println("Exemple 3")
t=0.0;s=2.0;r=2.0;
nlp_ma3=ADNLPModel(x->mpcc.mp.f(x)+x[3]^2+x[4]^2, [1;0;s;3],lvar=[-Inf;-Inf;-r;-r],uvar=ones(4)*Inf)
ma3=ActifMPCCmod.MPCC_actif(nlp_ma3,r,s,t,1)

if !(ma3.w[2,1]==1 && ActifMPCCmod.evalx(ma3,[1;0;3])==[1;0;s;3] && ActifMPCCmod.obj(ma3,[1;0;3])==14 && ActifMPCCmod.grad(ma3,[1;0;3])==[1;-1;6])
 println("Exemple 3")
 println(ma3.w) # (2,1)=1
 println(ActifMPCCmod.evalx(ma3,[1;0;3])) #doit rentre la même chose que xkf
 println(ActifMPCCmod.obj(ma3,[1;0;3])) #doit donner xkf[1]-xkf[2]+xkf[3]^2+xkf[4]^2=14
 #gradient : [1;-1;2x3;2x4]=[1;-1;4;6]
 println(ActifMPCCmod.grad(ma3,[1;0;3])) #doit donner [1,-1,6]

 ActifMPCCmod_success=false
end

#println("Exemple 4")
t=0.0;s=2.0;r=2.0;
nlp_ma4=ADNLPModel(x->mpcc.mp.f(x)+x[3]^2+x[4]^2, [1;0;0;s],lvar=[-Inf;-Inf;-r;-r],uvar=ones(4)*Inf)
ma4=ActifMPCCmod.MPCC_actif(nlp_ma4,r,s,t,1)
xkf=[1;0;0;s]
if !(ma4.w[2,2]==1 && ActifMPCCmod.evalx(ma4,[1;0;0])==xkf && norm(ActifMPCCmod.obj(ma4,[1;0;0])-5)<precision && ActifMPCCmod.grad(ma4,[1;0;0])==[1;-1;0])
 println("Exemple 4")
 println(ma4.w) # (2,2)=1
 println(ActifMPCCmod.evalx(ma4,[1;0;0])) #doit rentre la même chose que xkf
 println(ActifMPCCmod.obj(ma4,[1;0;0])) #doit donner xkf[1]-xkf[2]+xkf[3]^2+xkf[4]^2=5
 #gradient : [1;-1;2x3;2x4]=[1;-1;0;4]
 println(ActifMPCCmod.grad(ma4,[1;0;0])) #doit donner [1,-1,0]

 ActifMPCCmod_success=false
end

#Bilan : donner une sortie bilan
if ActifMPCCmod_success==true
 println("ActifMPCCmod.jl passes the test !")
else
 println("ActifMPCCmod.jl contains some error")
end
