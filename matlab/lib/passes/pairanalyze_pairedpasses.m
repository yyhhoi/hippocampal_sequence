function pairout = pairanalyze_pairedpasses(pairedpasses)
    % Modified method

    minspikepairs=16;
    
    pairout.ok=0;
    stimesA=[];
    anglesA=[];
    vsA=[];
    stimesB=[];
    anglesB=[];
    vsB=[];
    directionsA = [];
    directionsB = [];
    for np=1:length(pairedpasses)
        % Stack directions
        direction = find_direction(pairedpasses(np).infield1, pairedpasses(np).infield2);  % 0=A->B, 1=B->A
        directionsA = [directionsA direction*ones(size(pairedpasses(np).tsp1'))];
        directionsB = [directionsB direction*ones(size(pairedpasses(np).tsp2'))];
        
        % Stack spiketimes and angles
        stimesA=[stimesA pairedpasses(np).tsp1'];
        anglesA=[anglesA pairedpasses(np).spike1angle];
        stimesB=[stimesB pairedpasses(np).tsp2'];
        anglesB=[anglesB pairedpasses(np).spike2angle];
        
        % Stack straightness
        [meanangle, nspikes1, nspikes2, straightR1, straightR2] = straightrank_each(pairedpasses(np));
        vsA=[vsA straightR1*ones(size(pairedpasses(np).tsp1'))];
        vsB=[vsB straightR2*ones(size(pairedpasses(np).tsp2'))];
    end
  
    if length(stimesA)*length(stimesB)==0
        isi=[];
    else
        isi = stimesA'*ones(size(stimesB)) - ones(size(stimesA'))*stimesB;
    end
    id0=find(abs(isi(1:end))<.15);
    spikepairs=length(id0);
    
    if sum(spikepairs) < minspikepairs
        pairout.phi0=nan;
        pairout.phi1=nan;
        pairout.theta0=nan;
        pairout.phase_AB = nan;
        pairout.phase_BA = nan;
    else
        pairout.ok=sum(spikepairs);
        % Find theta period
        [h b]=hist(isi(id0),-.15:.005:.15);
        h0=h-mean(h);
        fs=1/(b(2)-b(1));
        wl=7/(fs/2);
        wh=10/(fs/2);
        [Bb,Ab]=butter(4,[wl wh]);
        z=hilbert(filtfilt(Bb,Ab,h));
        alpha=angle(z);
        idperiod=find(alpha(2:end)-alpha(1:end-1)<-pi);
        Tperiod(1)=1/9;% default theta
        for ni=2:length(idperiod)
            Tperiod(ni-1)=b(idperiod(ni))-b(idperiod(ni-1));
        end
        Ttheta=mean(Tperiod);
        clear Tperiod
        
        % Stack for doubleCircularFit
        phiC=[];
        thetaC=[];
        phiC_AB = [];
        phiC_BA = [];
        thetaC_AB=[];
        thetaC_BA=[];
        
        for na=1:length(stimesA)
            ta=stimesA(na);
            idin=find(abs(ta-stimesB)<.08);
            dt=stimesB(idin)-ta;
            phi=mod(dt,Ttheta)/Ttheta*.2;
            phi(phi>.1)=phi(phi>.1)-.2;
            if ~isnan(vsA(na)) & length(phi)
                phiC=[phiC phi/.1*pi];
                thetaC=[thetaC anglesB(idin)];
            end
            
            if directionsA(na) == 0
                phiC_AB = [phiC_AB phi/.1*pi];
                thetaC_AB=[thetaC_AB anglesB(idin)];
            elseif directionsA(na) == 1
                phiC_BA = [phiC_BA phi/.1*pi];
                thetaC_BA=[thetaC_BA anglesB(idin)];
            end
            
        end

        for na=1:length(stimesB)
            ta=stimesB(na);
            idin=find(abs(ta-stimesA)<.08); 
            dt=stimesA(idin)-ta;
            phi=mod(dt,Ttheta)/Ttheta*.2;
            phi(phi>.1)=phi(phi>.1)-.2;
            phi=-phi;

            if ~isnan(vsB(na)) & length(phi)
                phiC=[phiC phi/.1*pi];
                thetaC=[thetaC anglesA(idin)];
            end
            
            if directionsB(na) == 0
                phiC_AB = [phiC_AB phi/.1*pi];
                thetaC_AB=[thetaC_AB anglesA(idin)];
            elseif directionsB(na) == 1
                phiC_BA = [phiC_BA phi/.1*pi];
                thetaC_BA=[thetaC_BA anglesA(idin)];
            end

        end
        
        
        
        [phi0, phi1, theta0, ost]=doublecircfit(thetaC, phiC);
        % Smoothening
        sig= pi/8; 
        gauss = @(x, mu) exp(-((x - mu).^2)/(2*sig^2))/(sqrt(2*pi)*sig);
        x_arr = -pi:0.01:pi;
        
        if length(phiC_AB) < 1
            phase_AB = nan;
        else
            phiC_AB_smooth = zeros(size(x_arr));
            for idx_AB=1:length(phiC_AB)
                phiC_AB_smooth = phiC_AB_smooth + gauss(x_arr, phiC_AB(idx_AB));
            end
            phase_AB = x_arr(find(phiC_AB_smooth == max(phiC_AB_smooth)));
        end
       
        if length(phiC_BA) < 1
            phase_BA = nan;
        else
            phiC_BA_smooth = zeros(size(x_arr));
            for idx_BA=1:length(phiC_BA)
                phiC_BA_smooth = phiC_BA_smooth + gauss(x_arr, phiC_BA(idx_BA));
            end
            phase_BA = x_arr(find(phiC_BA_smooth == max(phiC_BA_smooth))); 
        end
        
            
        pairout.phase_AB = phase_AB;
        pairout.phase_BA = phase_BA;
        pairout.phi0=phi0;
        pairout.phi1=phi1;
        pairout.theta0=theta0;
        pairout.ost=ost;
        pairout.thetaC = thetaC;
        pairout.phiC = phiC;

        pairout.occ=tuning(angle(exp(i*([anglesA anglesB]-theta0))),6);
        tout=tuning(angle(exp(i*(thetaC-theta0))),6);
        pairout.tuning=correctprior(tout,pairout.occ);
        
    end
end