module OutputALASmod

using OutputLSmod #naming problem (Print)

type OutputALAS

 solved                  :: Int64 #renvoi 0 si réussi
 solve_message           :: AbstractString #message de sortie
 xtab                    :: Array #tableau des itérés
 dtab                    :: Array #tableau des directions
 rhotab                  :: Array{Float64,1}
 steptab                 :: Array{Float64,1}

 realisabilite           :: Array{Float64,1}
 dualrealisabilite       :: Array{Float64,1}
 obj                     :: Array{Float64,1}

 inner_output_linesearch ::Any #en fait une liste de OutputLS

end

#Initialisation
function OutputALAS(x0       :: Vector,
                    dj       :: Vector,
                    feas     :: Float64,
                    dualfeas :: Float64,
                    rho      :: Vector,
                    ht       :: Float64)

 return OutputALAS(0,"Success",x0,dj,[norm(rho,Inf)],[],[feas],[dualfeas],[ht],[])
end

#Mise à jour
#à faire autrement !!
function oa_update!(oa       :: OutputALAS,
                    xk       :: Vector,
                    rho      :: Vector,
                    feas     :: Float64,
                    dualfeas :: Float64,
                    dj       :: Vector,
                    step     :: Float64,
                    outputls :: OutputLSmod.OutputLS,
                    ht       :: Float64,
                    n        :: Int64)

 oa.realisabilite = [oa.realisabilite; feas]
 oa.dualrealisabilite = [oa.dualrealisabilite; dualfeas]
 oa.obj = [oa.obj; ht]
 oa.rhotab = [oa.rhotab; norm(rho,Inf)]
 oa.steptab = [oa.steptab; step]
 oa.xtab = [oa.xtab xk]
 oa.dtab = [oa.dtab dj]
 oa.inner_output_linesearch = [oa.inner_output_linesearch; outputls]

 Print(oa, n, 2, j=length(oa.steptab))

 return oa
end

#Print
function Print(oa      :: OutputALAS,
               n       :: Int64,
               verbose :: Int64;
               j       :: Int64=1)

 if verbose>1
  ncc=Int64(0.5*(length(oa.xtab)/length(oa.realisabilite)-n))
  while j <= length(oa.steptab)
   #println("k :",j," xjk=",oa.xtab[1:n,j+1]," sg=",oa.xtab[n+1:n+ncc,j+1]," sh=",oa.xtab[n+ncc+1:n+2*ncc,j+1],"\n\|c(xjk)\|=",oa.realisabilite[j+1]," JL(xjk)=",oa.dualrealisabilite[j+1]," rho= ",oa.rhotab[j+1]," \nalpha=",oa.steptab[j]," d=",oa.dtab[1:n+2*ncc,j+1])
   if n<=4
    if ncc>0
     #print_with_color(:blue, "	k: $j xjk=$(oa.xtab[1:n,j+1]) sg=$(oa.xtab[n+1:n+ncc,j+1]) sh=$(oa.xtab[n+ncc+1:n+2*ncc,j+1]) \n\|c(xjk)\|=$(oa.realisabilite[j+1]) JL(xjk)=$(oa.dualrealisabilite[j+1]) rho=$(oa.rhotab[j+1]) \nalpha=$(oa.steptab[j]) d=$(oa.dtab[1:n+2*ncc,j+1]) fpen=$(oa.obj[j]) \n")
     print_with_color(:blue, "	k: $j \|xjk\|=$(norm(oa.xtab[1:n,j+1],Inf)) \|c(xjk)\|=$(oa.realisabilite[j+1]) L'(xjk)=$(oa.dualrealisabilite[j+1]) rho=$(oa.rhotab[j+1]) \nalpha=$(oa.steptab[j]) \|d\|=$(norm(oa.dtab[1:n+2*ncc,j+1],Inf)) fpen=$(oa.obj[j]) \n")
    else     
     print_with_color(:blue, "	k: $j xjk=$(oa.xtab[1:n,j+1]) \n\|c(xjk)\|=$(oa.realisabilite[j+1]) JL(xjk)=$(oa.dualrealisabilite[j+1]) rho=$(oa.rhotab[j+1]) \nalpha=$(oa.steptab[j]) d=$(oa.dtab[1:n+2*ncc,j+1]) fpen=$(oa.obj[j]) \n")
    end
   else
    if ncc>0
     #print_with_color(:blue, "	k: $j xjk=[$(oa.xtab[1,j+1])..$(oa.xtab[n,j+1])] sg=[$(oa.xtab[n+1,j+1])..$(oa.xtab[n+ncc,j+1])] sh=[$(oa.xtab[n+ncc+1,j+1])..$(oa.xtab[n+2*ncc,j+1])] \n\|c(xjk)\|=$(oa.realisabilite[j+1]) JL(xjk)=$(oa.dualrealisabilite[j+1]) rho=$(oa.rhotab[j+1]) \nalpha=$(oa.steptab[j]) d=(oa.dtab[1:n+2*ncc,j+1]) fpen=$(oa.obj[j]) \n")
     print_with_color(:blue, "	k: $j \|xjk\|=$(norm(oa.xtab[1:n,j+1],Inf)) \|c(xjk)\|=$(oa.realisabilite[j+1]) L'(xjk)=$(oa.dualrealisabilite[j+1]) rho=$(oa.rhotab[j+1]) \nalpha=$(oa.steptab[j]) \|d\|=$(norm(oa.dtab[1:n+2*ncc,j+1],Inf)) fpen=$(oa.obj[j]) \n")
    else     
     print_with_color(:blue, "	k: $j xjk=[$(oa.xtab[1,j+1])..$(oa.xtab[n,j+1])] \n\|c(xjk)\|=$(oa.realisabilite[j+1]) JL(xjk)=$(oa.dualrealisabilite[j+1]) rho=$(oa.rhotab[j+1]) \nalpha=$(oa.steptab[j]) d=(oa.dtab[1:n+2*ncc,j+1]) fpen=$(oa.obj[j]) \n")
    end
   end
   OutputLSmod.Print(oa.inner_output_linesearch[j],n,verbose)
   j+=1
  end
  println("")
 end

 return true
end

#end of module
end
