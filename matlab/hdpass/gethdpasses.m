function hdinfo=gethdpasses(passes,thresh)

  vthresh=5;
  NShuffles=200;
  % 
  %
  allangles=[];
  allspikeangles=[];
  alldt=[];
  %ok(length(passes)).lst=[];
  nok=0;
  for np=1:length(passes)
    %
    idok=find(passes(np).v>vthresh);
    dt=passes(np).t(2:end)-passes(np).t(1:end-1);
    dt(end+1)=dt(end);
    
    
    %straightness criterion meanR(np)^2 > (1+thresh*sqrt(1-1/ntmp))/ntmp
    straightthresh(np)=((abs(mean(exp(i*passes(np).angle(idok))))^2)*length(idok)-1)/sqrt(1-1/length(idok));
%    apass(np).a=nan;
    
    if straightthresh(np)>thresh
      alldt=[alldt dt(idok)'];
      allangles=[allangles passes(np).angle(idok)'];
      %ok(np).lst=idok;
      
      idoksp=find(passes(np).spikev>vthresh);
      allspikeangles=[allspikeangles passes(np).spikeangle(idoksp)];
      nok=nok+sum(~isnan(passes(np).spikeangle(idoksp)));
      
 %     apass(np).a=angle(sum(exp(i*passes(np).angle(idok))));
    end
  end
  abins=-pi:2*pi/36:pi;
  acntrs=(abins(1:end-1)+abins(2:end))/2;
  [odummy, adummy, dict]=histcounts(allangles,abins);
  for nbin=1:length(odummy)
    if length(dict)>0
      occupancy(nbin)=sum(alldt(find(dict==nbin)));
    else
      occupancy(nbin)=nan;
    end
  end
  
  scnt=histcounts(allspikeangles,abins);
  
  fprob=scnt./occupancy;
  
  hdinfo.fprob=fprob;
  hdinfo.RVL=abs(sum(fprob.*exp(i*acntrs)))/sum(fprob);
  hdinfo.N=nok;
  hdinfo.occupancy=occupancy;
  
  %%%
  %%% NULL Hypothesis
  %%%

  for ns=1:NShuffles
    spikeashuff=[];
    
    pidshuff=randperm(length(passes));
    for np=1:length(passes)
      if straightthresh(np)>thresh
        %%%% Change begins here  %%%%
        pass_angles = passes(np).angle;
        t = passes(np).t;
        tsp = passes(np).tsp;
        if length(tsp) < 1
            continue
        end
        if length(pass_angles) == length(t)
            time_length = t(end) - t(1);
            rand_start = t(1) + rand * time_length;
            new_tsp = tsp - tsp(1) + rand_start;
            new_tsp = mod(new_tsp - t(1), time_length) + t(1);  % Circular shift
            shifted_spike_angles = interp1(t, pass_angles, new_tsp);
            spikeashuff = [spikeashuff shifted_spike_angles'];
        end          
        %%%% Change ends here  %%%%
      end
    end
    
    
    scntshuff=histcounts(spikeashuff,abins);
    clear spikeashuff;
    
    fprobshuff=scntshuff./occupancy;    
    RVLshuff(ns)=abs(sum(fprobshuff.*exp(i*acntrs)))/sum(fprobshuff);
    
  end
  
  crit=hdinfo.RVL>RVLshuff;
  hdinfo.pval=1-sum(crit)/NShuffles;
  
