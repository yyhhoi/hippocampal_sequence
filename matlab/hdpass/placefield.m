function pf = placefield(xyt,xytsp)
  
  sig= 3.;% in cm
  gauss = @(x,y) exp(-(x.^2+y.^2)/(2*sig^2))/(2*pi*sig^2);
  
  xmin=min(xyt.x);
  xmax=max(xyt.x);
  ymin=min(xyt.y);
  ymax=max(xyt.y);
  
  xmin=-max(abs([xmin xmax]));
  xmax=max(abs([xmin xmax]));

  ymin=-max(abs([ymin ymax]));
  ymax=max(abs([ymin ymax]));

  tmpx=-1:-1:xmin ;
  tmpy=-1:-1:ymin ;
  [X,Y] = meshgrid([tmpx(end:-1:1) 0:1:xmax], [tmpy(end:-1:1) 0:1:ymax]);
  occupancy=zeros(size(X));  
  
  %radius=max(max(sqrt(xyt.x.^2+xyt.y.^2)));  
  
  
  dt=xyt.t(2:end)-xyt.t(1:end-1);
  chunks=find(dt>.3);
  chunks(end+1)=length(xyt.t);
  iold=1;
  
  for nc=1:length(chunks)
    
    tarr=xyt.t(iold:chunks(nc));

    
  
    for nt=1:length(tarr)-1
      dt=tarr(nt+1)-tarr(nt);
      it=iold+nt-1;
      occupancy=occupancy+gauss(xyt.x(it)-X,xyt.y(it)-Y)*dt;
    end
    
    iold=chunks(nc)+1;
    if iold > length(xyt.t)
      continue;
    end
  end

  
  rates=zeros(size(X));
  for nt=1:length(xytsp.t)
    rates=rates+gauss(xytsp.x(nt)-X,xytsp.y(nt)-Y);
  end
  
  
  pf.map=rates./(0.0001+occupancy);
  pf.X=X;
  pf.Y=Y;
  pf.occ=occupancy/sum(sum(occupancy));
  pf.rates=rates;
  