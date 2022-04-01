function hdinfo=gethd(field,idat,xytsp)

  vthresh=5;
  NShuffles=200;
  %
  
  mask=1.*field.mask;
  mask(field.mask==0)=nan;
    
  xax=field.pf.X(1,:);
  yax=field.pf.Y(:,1)';
  dmatx=((idat.x*ones(size(xax))-ones(size(idat.x))*xax)).^2;
  dmaty=((idat.y*ones(size(yax))-ones(size(idat.y))*yax)).^2;
  [dum idx]=min(dmatx');
  [dum idy]=min(dmaty');
 

  for nstep=1:length(idy)
    
    tok(nstep)=mask(idy(nstep),idx(nstep));
  end
  
  dx=conv(idat.x(2:end)-idat.x(1:end-1),[0 1 1 1]/3,'same');
  dy=conv(idat.y(2:end)-idat.y(1:end-1),[0 1 1 1]/3,'same');;
  dt=conv(idat.t(2:end)-idat.t(1:end-1),[0 1 1 1]/3,'same');;
  velocity=sqrt((dx.^2 + dy.^2))./dt;  
  movedir=angle(dx+i*dy);


  idin=find(~isnan(tok(1:end-1)) & velocity'>vthresh);
  abins=-pi:2*pi/36:pi;
  acntrs=(abins(1:end-1)+abins(2:end))/2;
  [odummy, adummy, dict]=histcounts(movedir(idin),abins);
  for nbin=1:length(odummy)
    occupancy(nbin)=sum(dt(idin(find(dict==nbin))));
  end
  
  nok=0;
  for nsp=1:length(xytsp.t)
    spiketime=xytsp.t(nsp);
    [dum,ids]=min(abs(spiketime-idat.t(1:end-1)));
    
    i0=find(ids==idin);
    
    if length(i0)>0 
      isp=idin(i0);

      if isp+1<=length(idat.t)-1
        nok=nok+1;

        
        tt0=idat.t(isp);
        %x0=idat.x(isp);
        %y0=idat.y(isp);
        a0=movedir(isp);
        
        tt1=idat.t(isp+1);
        %x1=idat.x(isp+1);
        %y1=idat.y(isp+1);
        a1=movedir(isp+1);
        %spikex(nok)=x0+(x1-x0)/(tt1-tt0)*(spiketime-tt0);
        %spikey(nok)=y0+(y1-y0)/(tt1-tt0)*(spiketime-tt0);
        %spiket(nok)=spiketime;
        da=angle(exp(i*(a1-a0)));
        spikea(nok)=angle(exp(i*(a0+da*(spiketime-tt0)/(tt1-tt0))));%linear interpolation
        
      end
    end
  
  end
    

  scnt=histcounts(spikea,abins);
  
  fprob=scnt./occupancy;
  
  hdinfo.fprob=fprob;
  hdinfo.RVL=abs(sum(fprob.*exp(i*acntrs)))/sum(fprob);
  hdinfo.N=nok;
  hdinfo.occupancy=occupancy;
  
  %%%
  %%% NULL Hypothesis
  %%%
  for ns=1:NShuffles
    s0=floor(rand*length(idin));
    cycshift=mod(s0+(1:length(idin)),length(idin));
    cycshift(cycshift==0)=length(idin);
    jdin=idin(cycshift);
    
    nok=0;
    for nsp=1:length(xytsp.t)
      spiketime=xytsp.t(nsp);
      [dum,ids]=min(abs(spiketime-idat.t(1:end-1)));
      
      i0=find(ids==idin);
      
      if length(i0)  
        jsp=jdin(i0);
        isp=idin(i0);
      
        if jsp+1<=length(idat.t)-1
          nok=nok+1;
          srel=spiketime-idat.t(isp)+idat.t(jsp);
          
          tt0=idat.t(jsp);
          %x0=idat.x(isp);
          %y0=idat.y(isp);
          a0=movedir(jsp);
          
          tt1=idat.t(jsp+1);
          %x1=idat.x(isp+1);
          %y1=idat.y(isp+1);
          a1=movedir(jsp+1);
          %spikex(nok)=x0+(x1-x0)/(tt1-tt0)*(spiketime-tt0);
          %spikey(nok)=y0+(y1-y0)/(tt1-tt0)*(spiketime-tt0);
          %spiket(nok)=spiketime;
          da=angle(exp(i*(a1-a0)));
          spikeashuff(nok)=angle(exp(i*(a0+da*(srel-tt0)/(tt1-tt0))));%linear interpolation
        
        end
      end
  
    
  
    end
    scntshuff=histcounts(spikeashuff,abins);
    clear spikeashuff;
    
    fprobshuff=scntshuff./occupancy;    
    RVLshuff(ns)=abs(sum(fprobshuff.*exp(i*acntrs)))/sum(fprobshuff);
    
  end
  
  crit=hdinfo.RVL>RVLshuff;
  hdinfo.pval=1-sum(crit)/NShuffles;
  
  hdinfo.peakrate=field.peakrate;
  hdinfo.area=field.area;