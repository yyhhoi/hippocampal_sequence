function tmppass=shufflepass(spikepass,runpass)
  

  tmppass.spikev=[];
  tmppass.spikeangle=[];
  
  trun=runpass.t-runpass.t(1);
  if length(spikepass.t)>0
    tspike=spikepass.tsp-spikepass.t(1);
    tspike=mod(tspike,trun(end)+.25);
  end
  
  for nsp=1:length(spikepass.tsp)
    
    i0=min(find(tspike(nsp)>=trun));
    t0=trun(i0);
    
    if i0+1<=length(trun)
      t1=trun(i0+1);
      xi=(tspike(nsp)-t0)/(t1-t0);
      
      da=angle(exp(i*xi*(runpass.angle(i0+1)-runpass.angle(i0))));
      
      tmppass.spikeangle(nsp)=angle(exp(i*(runpass.angle(i0)+da)));

      tmppass.spikev(nsp)=runpass.v(i0) + xi*(runpass.v(i0+1)-runpass.v(i0));
    else
      
      tmppass.spikeangle(nsp)=runpass.angle(i0);
      tmppass.spikev(nsp)=runpass.v(i0);
    end
  end