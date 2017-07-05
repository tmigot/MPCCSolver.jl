function [resultat, n_pb, n_pb_solved ] = Theta_MacMPEC_run(t0,r0,sigma_r,sigma_t,tol_vio,tol_param,presolve,solver,name)

%{
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Tangi : 05/09 bug sur les codes erreurs >400 rectifié.
% Tangi (12/09) : ajout d'un vecteur 3x1 pour tol_vio
%		- variable maxVio <-> maxVioComp
		- nouvelles variables : feas, comp, lagr
		- ajout d'un timer maximum
A faire :
	- resultat <- structure de donnée plutôt qu'un tableau
	- traitement plus fin sur les codes de retour (voir description dans le fichier Comparaison)
	- revoir le critère d'arrêt.
%}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% paramètres :
max_time=60;
if min(size(tol_vio)==[1 1])
	tol_cons=tol_vio;
	tol_comp=tol_vio;
	tol_lagr=tol_vio;
elseif min(size(tol_vio)==[3 1]) || min(size(tol_vio)==[1 3])
	tol_cons=tol_vio(1);
	tol_comp=tol_vio(2);
	tol_lagr=tol_vio(3);
else
	% tol_vio error dimension must be (1,1), (3,1) or (1,3)
	return;
end
% on demande min(G,H)<=tol_demande et GH<=tol_cons.
tol_demande=sqrt(tol_cons);

% Create an AMPL instance
ampl = AMPL;
% Display version
ampl.eval('option version;')

% Initialisation
basef = fileparts(which('bard1Model'));
modeldirectory = fullfile(basef,[name,'models']);

% initialisation des variables :
resultat=[];stat='0';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Les exemples qui posent problème sont le fichier NotesTestMPEC.ods.
% le 1er groupe :
% bilevel1m, bilevel2m : à voir comment ça marche.
name_pb_1={'bard1','bard1m','bard2','bard3','bard3m','bar-truss','bilevel1','bilevel2','bilevel3','bilin','dempe','desilva','df1','flp2','flp4','flp4','gauvin','hakonsen','hs044-i'};
data_pb_1={'','','','','','bar-truss-3','','','','','','','','','flp4-1','flp4-2','','',''};
f_mpec_1=[17,17,-6598.00,-12.6787 ,-12.6787, 10166.6,0 ,-6600.00,-12.6787 ,-18.4 ,28.25 ,-1.0,0.0,0,0,0,20.0,-24.3668,15.6178];

% les problèmes design :
name_pb_design={'design-cent-1','design-cent-2','design-cent-21','design-cent-3','design-cent-31','design-cent-4'};
data_pb_design={'design-cent-1','design-cent-2','design-cent-2','design-cent-3','design-cent-3','design-cent-4'};
f_mpec_design=[-1.86065,-3.48382,-3.48382,-3.7237,-3.72337,-3.0792];

%le groupe des ex
name_pb_ex={'ex9.1.1','ex9.1.2','ex9.1.3','ex9.1.4','ex9.1.5','ex9.1.6','ex9.1.7','ex9.1.8','ex9.1.9','ex9.1.10','ex9.2.1','ex9.2.2','ex9.2.3','ex9.2.4','ex9.2.5','ex9.2.6','ex9.2.7','ex9.2.8','ex9.2.9'};
data_pb_ex={'','','','','','','','','','','','','','','','','','',''};
f_mpec_ex=[-13.0 ,-6.25,-29.2,-37.0,-1.0, -49 ,-26.0,-3.25,3.11111,-3.25 ,17.0,100.0,-55,0.5,5.0 ,-1.0 ,17.0,1.5 ,2.0];

%le groupe nash
%name_pb_nash={'gnash1-m','gnash1-m','gnash1-m','gnash1-m','gnash1-m','gnash1-m','gnash1-m','gnash1-m','gnash1-m','gnash1-m','gnash1','gnash1','gnash1','gnash1','gnash1','gnash1','gnash1','gnash1','gnash1','gnash1','nash1','nash1','nash1','nash1','nash1'};
%data_pb_nash={'gnash10','gnash11','gnash12','gnash13','gnash14','gnash15','gnash16','gnash17','gnash18','gnash19',...
%'gnash10','gnash11','gnash12','gnash13','gnash14','gnash15','gnash16','gnash17','gnash18','gnash19',...
%'nash1a','nash1b','nash1c','nash1d','nash1e'};
%f_mpec_nash=[-230.823,-129.912 ,-36.9331 ,-7.06178 ,-0.179046,-354.699 ,-241.442,-90.7491,-25.6982,-6.11671  ,...
%-230.823 ,-129.912 ,-36.9331 ,-7.06178 ,-0.179046,-354.699 ,-241.442,-90.7491,-25.6982,-6.11671,...
%7.89e-030, 7.89e-030, 7.89e-030, 7.89e-030, 7.89e-030];
name_pb_nash={'nash1','nash1','nash1','nash1','nash1'};
data_pb_nash={'nash1a','nash1b','nash1c','nash1d','nash1e'};
f_mpec_nash=[7.89e-030, 7.89e-030, 7.89e-030, 7.89e-030, 7.89e-030];

%Le 2ème groupe:
% monteiro et monteiroB à voir comment ça marche
name_pb_2={'incid-set1','incid-set1c','incid-set2','incid-set2c','jr1','jr2','kth1','kth2','kth3','liswet1-inv','outrata31','outrata32','outrata33','outrata34','portfl-i','portfl-i','portfl-i','portfl-i','portfl-i'};
data_pb_2={'incid-set-8','incid-set-8','incid-set-8','incid-set-8','','','','','','liswet1-050','','','','','portfl1','portfl2','portfl3','portfl4','portfl6'};
f_mpec_2=[3.82e-017,3.82e-017,4.52e-003,5.47e-003,0.5 ,0.5 ,0 ,0,0.5,1.40E-002,3.2077 ,3.4494 ,4.60425,6.59268,1.502E-5, 1.457E-5, 6.265E-6, 2.177E-6, 2.361E-6];

%le groupe des packs
name_pb_pack={'pack-comp1','pack-comp1c','pack-comp1p','pack-comp2','pack-comp2c','pack-comp2p','pack-rig1','pack-rig1c','pack-rig1p','pack-rig2','pack-rig2c','pack-rig2p','pack-rig3','pack-rig3c'};
data_pb_pack={'pack-comp-8','pack-comp-8','pack-comp-8','pack-comp-8','pack-comp-8','pack-comp-8','pack-rig-8','pack-rig-8','pack-rig-8','pack-rig-8','pack-rig-8','pack-rig-8','pack-rig-8','pack-rig-8'};
f_mpec_pack=[0.6 ,0.6,0.6 ,0.673117,0.6734580,0.673117,0.787932 ,0.7883 ,0.787932,0.780404 ,0.799306,0.780404, 0.735202, 0.753473];

%Le 3ème groupe :
% taxmcp, TrafficSignalCycle à voir comment ça marche
name_pb_3={'qpec1','qpec2','ralph1','ralph2','scholtes1','scholtes2','scholtes3','scholtes4','scholtes5',...
'scale1','scale2','scale3','scale4','scale5','sl1','stackelberg1','tap-09','water-net','water-net'};
data_pb_3={'','','','','','','','','','','','','','','','','tap-09','water-net','water-FL'};
f_mpec_3=[80, 45.0,0,0,2,15,0.5,-3.07336e-7,1,  1.0 ,1.0 ,1.0,1.0  ,100.0,1e-4,-3266.67,109.143, 929.169, 3411.92];

name_pb=[name_pb_1,name_pb_design,name_pb_ex,name_pb_nash,name_pb_2, name_pb_pack, name_pb_3];
data_pb=[data_pb_1,data_pb_design,data_pb_ex,data_pb_nash,data_pb_2, data_pb_pack, data_pb_3];
f_mpec=[f_mpec_1,f_mpec_design,f_mpec_ex,f_mpec_nash,f_mpec_2, f_mpec_pack, f_mpec_3];

name_pb=[name_pb_1,name_pb_design,name_pb_ex,name_pb_2, name_pb_pack, name_pb_3];
data_pb=[data_pb_1,data_pb_design,data_pb_ex,data_pb_2, data_pb_pack, data_pb_3];
f_mpec=[f_mpec_1,f_mpec_design,f_mpec_ex,f_mpec_2, f_mpec_pack, f_mpec_3];

%name_pb=[name_pb_1,name_pb_ex,name_pb_nash,name_pb_2, name_pb_pack, name_pb_3];
%data_pb=[data_pb_1,data_pb_ex,data_pb_nash,data_pb_2, data_pb_pack, data_pb_3];
%f_mpec=[f_mpec_1,f_mpec_ex,f_mpec_nash,f_mpec_2, f_mpec_pack, f_mpec_3];

%name_pb_tangi={'GLY2.1','GLY2.2','GLY2.3','GLY2.4'};
%data_pb_tangi={'','','',''};
%f_mpec_tangi=[0,0,5/4,5/4];

name_pb={'Tangi5'};
data_pb={''};
f_mpec=[exp(-100)];

%{
name_pb={'GLY2.2'};
data_pb={''};
f_mpec=[0];
%}
%{
name_pb={'GLY2.4'};
data_pb={''};
f_mpec=[5/4];
%}
%}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if size(f_mpec,2)~=size(data_pb,2) || size(name_pb,2)~=size(data_pb,2) || size(f_mpec,2)~=size(name_pb,2)
	disp('Data error : size of [f_mpec,name_pb,data_pb] not consistent');
	size(f_mpec,2)
	size(data_pb,2)
	size(name_pb,2)
else

%legend_t={'nom probleme','data','f_mpec','f_opt','maxVio','tfin','rfin','strict comp.','réussite','stat.','solve'};
%resultat=[resultat;legend_t(1),legend_t(3),legend_t(4),legend_t(5),legend_t(6),legend_t(7),legend_t(8),legend_t(9),legend_t(10),legend_t(11)];

n_pb_solved=0;output=[];
n_pb=size(name_pb,2)

for i=1:n_pb
	tic;mtime=0;
	ampl.eval('reset;');
	% Load from file the ampl model
	ampl.read([fullfile(modeldirectory,strcat(name_pb(i),[name,'.mod']))]);
	ampl.readData([fullfile(modeldirectory,strcat(data_pb(i),'.dat'))]);
	% valeurs par défaut
	t=t0;r=r0;scomp=NaN;maxVioComp=NaN;lagr(i)=NaN;feas(i)=NaN;comp(i)=NaN;sd=NaN;sc=NaN;fopt=Inf;

		% choix du point initial :
	%graine=1;
	%ampl.eval(strcat('option randseed ',{' '}, num2str(graine),';'));
	ampl.eval('let x:= 0.1;');
	ampl.eval('let y:= 0.5;');
	ampl.eval('let z:= 0;');
	ampl.eval('let w:= 0;');

	ampl.eval('param rap=solve_result_num;')
	ampl.eval('param fobj;')
	ampl.eval('param maxVio;')
	ampl.eval('param maxCom;')
	ampl.eval('param maxLag;')
	ampl.eval('param sd;')
	ampl.eval('param sc;')
	% au cas où il y ait un problème avec l'évaluation des contraintes :
	ampl.eval('let maxVio := -1 ;');ampl.eval('let maxCom := -1 ;');
name_pb(i)
data_pb(i)
	while max(t,r)>tol_param && ((t==t0 && r==r0) || lagr(i)>tol_lagr || comp(i)>tol_comp || feas(i)>tol_cons || maxVioComp>tol_demande ) && mtime<=max_time 
		%ampl.readData([modeldirectory  '/' 'param.dat']);
		ampl.getParameter('r').setValues(r);
		ampl.getParameter('t').setValues(t);
		% Solve
		ampl.setOption('solver_msg','0');
		% par défaut : presolve 10
		ampl.setOption('presolve',num2str(presolve));
		ampl.setOption('show_stats','0');
		ampl.setOption('solver' ,solver);
		ampl.setOption('display_precision' ,'10');
		ampl.setOption('solution_round' ,'15');
		% autre option : scale=yes/no
		if strcmp(solver,'kestrel')
			ampl.eval('option kestrel_options ''solver=loqo''; ');
			%ampl.eval('option kestrel_options ''solver=knitro''; ');
		elseif strcmp(solver,'snopt')
			%tau=1e-2;
			%ampl.eval(strcat('option snopt_options ''feasibility_tolerance=',num2str(min(tau,max(t,r))),''';'));
			ampl.eval(strcat('option snopt_options ''Major_feasibility_tolerance=',num2str(tol_cons),' Major_optimality_tolerance=',num2str(tol_comp),' Minor_feasibility_tolerance=',num2str(tol_comp),' '';'));
		elseif strcmp(solver,'minos')
			%tau=1e-2;
			%ampl.eval(strcat('option minos_options ''feasibility_tolerance=',num2str(min(tau,max(t,r))),''';'));
			ampl.eval(strcat('option minos_options ''feasibility_tolerance=',num2str(tol_comp),' row_tolerance=',num2str(tol_cons),' optimality_tolerance=',num2str(tol_lagr),''';'));
		elseif strcmp(solver,'ipopt')
			%tau=1e-2;
			%ampl.eval(strcat('option ipopt_options "print_level=0 constr_viol_tol=',num2str(min(tau,max(sqrt(t),sqrt(r)))),'";'));
			ampl.eval(strcat('option ipopt_options ''print_level=0 compl_inf_tol=',num2str(tol_comp),' constr_viol_tol=',num2str(tol_cons),' dual_inf_tol=',num2str(tol_lagr),' acceptable_tol=',num2str(tol_lagr),' '';'));
		end
		ampl.solve

		% Print it
		G = ampl.getVariable('G').getValues().getColumnAsDoubles('val');
		H = ampl.getVariable('H').getValues().getColumnAsDoubles('val');
		mtime=toc;
		%names_cons = ampl.getConstraints();
		%names_cons
		%	for c=1:size(names_cons)
		%		names_cons(c);
		%	end
		% donne si le message de sortie du problème :
		output(i) = ampl.getParameter('rap').getValues().getColumnAsDoubles('val');
%		if (strcmp(solver,'minos') && (output(i)<=199 && output(i)>=0) || (output(i)<=501 && output(i)>=400)) || (strcmp(solver,'snopt') && (output(i)<=199 && output(i)>=0) || (output(i)<=501 && output(i)>=400)) || (strcmp(solver,'ipopt') && (output(i)<=199 && output(i)>=0) || (output(i)<=501 && output(i)>=400))
[output(i) t r]
		if (output(i)<=199 && output(i)>=0) || (output(i)<=401 && output(i)>=400) || (output(i)==520 && strcmp(solver,'minos'))
			%contrainte de complémentarité
			maxVioComp=norm(min(G,H),Inf);scomp=min(G+H);
			%critère de réalisabilité :
			ampl.eval('let maxVio := -min (0,min {i in 1.._ncons} _con[i].slack,min {j in 1.._nvars} _var[j].slack );');
			feas(i)=ampl.getParameter('maxVio').getValues().getColumnAsDoubles('val');
			% critère de complémentarité :
			ampl.display('max {i in 1.._ncons} min(max(_con[i].slack,-_con[i].slack),max(_con[i].dual,-_con[i].dual));');
			%ampl.eval('let maxCom := max {i in 1.._ncons} min(abs(_con[i].slack),abs(_con[i].dual));');
			% in case of non-strict complementarity :
			ampl.eval('let maxCom := max {i in 1.._ncons} min(min(abs(_con[i].slack),abs(_con[i].dual)),abs(_con[i].slack)*abs(_con[i].dual));');
			comp(i)=ampl.getParameter('maxCom').getValues().getColumnAsDoubles('val');
			% critère de dual réalisabilité (problème si pow(0,-0.75) pas différentiable) :
			try
			 if strcmp(solver,'ipopt')
			  %SPECIAL IPOPT :
			  ampl.eval(strcat('let maxLag := max {i in 1.._nvars} (if _var[i].slack==Infinity then abs(_var[i].rc) else _var[i].slack*abs(_var[i].rc));'));
			 else
			  ampl.eval(strcat('let maxLag := max {i in 1.._nvars} min(_var[i].slack,abs(_var[i].rc));'));
			 end
			 lagr(i)=ampl.getParameter('maxLag').getValues().getColumnAsDoubles('val');
			catch
			 lagr(i)=NaN;
			end
			%calcul des paramètres de scaling pour IPOPT
			ampl.eval('let sd := max(100.0,(sum {i in 1.._ncons} abs(_con[i].dual)+sum {i in 1.._nvars} abs(_var[i].rc))/(_ncons+_nvars))/100.0;');
			sd(i)=ampl.getParameter('sd').getValues().getColumnAsDoubles('val');
			ampl.eval('let sc := max(100.0,(sum {i in 1.._nvars} abs(_var[i].rc))/(_nvars))/100.0;');
			sc(i)=ampl.getParameter('sc').getValues().getColumnAsDoubles('val');
			% Get objective map by AMPL name
			try
			 ampl.eval('let fobj:=_obj[1];');
			 fopt=ampl.getParameter('fobj').getValues().getColumnAsDoubles('val');
			catch
			 fopt=Inf;
			end
		elseif output(i)==401 %iteration_limit

		else
			mtime=Inf;
		end

		t=t*sigma_t;
		r=r*sigma_r;
	end


	tfin=t/sigma_t;rfin=r/sigma_r;
%{
	if output(i)<600 && output(i)>=500 %fail
	reussite=Inf;
	resultat=[resultat; name_pb(i),f_mpec(i),Inf, tfin, rfin, Inf,feas(i), lagr(i),comp(i),output(i),reussite,stat, scomp,mtime];
	elseif output(i)<500 && output(i)>=400 %iteration limit
	reussite=Inf;
	resultat=[resultat; name_pb(i),f_mpec(i),Inf, tfin, rfin, Inf,feas(i), lagr(i),comp(i),output(i),reussite,stat, scomp,mtime];
	elseif output(i)<400 && output(i)>=300 %unbounded
	reussite=Inf;
	resultat=[resultat; name_pb(i),f_mpec(i),Inf, tfin, rfin,Inf, feas(i), lagr(i),comp(i),output(i),reussite,stat, scomp,mtime];
	elseif output(i)<300 && output(i)>=200 %infeasible
	reussite=Inf;
	resultat=[resultat; name_pb(i),f_mpec(i), Inf, tfin, rfin,Inf, feas(i), lagr(i),comp(i),output(i),reussite,stat, scomp,mtime];
	else %sortie entre 0 et 199
%}

		if min(G+H)>1e-3 || (t>tol_param && r>tol_param) || (~isempty(max(max(G(find(G+H<1e-3)),H(find(G+H<1e-3))))) && max(max(G(find(G+H<1e-3)),H(find(G+H<1e-3))))<min(t,r))
			stat='S';
		elseif r*r./(max(max(G(find(G+H<1e-3)),H(find(G+H<1e-3))))-t+r)<r
			stat='M';
		else
			stat='A';
		end

		reussite=abs(f_mpec(i)-fopt)/(1+abs(f_mpec(i)));

		resultat=[resultat; name_pb(i),f_mpec(i),fopt, tfin, rfin, maxVioComp, feas(i), lagr(i),comp(i),output(i),reussite,stat, scomp,mtime];
%	end
end

%debug display
if n_pb==1
	ampl.display('_varname,_var,_var.rc, _var.slack');
	ampl.display('_conname,_con.slack,_con.dual');
	tfin, rfin
end
			
% Close the AMPL object
ampl.close();
end
