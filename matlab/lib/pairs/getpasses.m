function passes=getpasses(field, xyt, xytsp)
% GETPASSES calculate one place field
%   passes = GETPASSES(field, xyt, xytsp) use structured data field, xyt
%   and xytsp to find multiple passes which were run through by the rat.
%   field: structured data of ONE place field. See getfields() function,
%   which gives output fields, of which the fields(i) will be the argument.
%   xyt: structured data with fields of x, y, t. All with (N, 1), N =
%   samples of positions.
%   xytsp: structured data with fields of x, y, t. All with (M, 1), M =
%   samples of spikes.

  minpasslength= 5*.1;% s
  
  mask=1.*field.mask;
  mask(field.mask==0)=nan;
  
  xc=sum(sum(field.pf.map.*field.pf.X))/sum(sum(field.pf.map));
  yc=sum(sum(field.pf.map.*field.pf.Y))/sum(sum(field.pf.map));
  drzmap=ones(size(field.pf.map))-field.pf.map/max(max(field.pf.map));
  
  xax=field.pf.X(1,:);
  yax=field.pf.Y(:,1)';
  dmatx=((xyt.x*ones(size(xax))-ones(size(xyt.x))*xax)).^2;
  dmaty=((xyt.y*ones(size(yax))-ones(size(xyt.y))*yax)).^2;
  [dum idx]=min(dmatx');
  [dum idy]=min(dmaty');
 
  passes=[];
  for nstep=1:length(idy)
    
    tok(nstep)=mask(idy(nstep),idx(nstep));
  end
  
  nstep=1;
  npass=0;
  while nstep<length(tok)
    trest=tok(nstep:end);
    
    if isnan(trest(1))
      i0=min(find(~isnan(trest)));
      if length(i0)==0
        break;
      end
      nstep=nstep+i0-1;
      trest=tok(nstep:end);
    end
    
    ie=min(find(isnan(trest)));
    if length(ie)==0
      ie=length(trest)+1;
    end
    tend=xyt.t(nstep+ie-2);
    tstart=xyt.t(nstep);

    if tend-tstart>minpasslength
      
      idsp=find(xytsp.t >= tstart & xytsp.t <= tend);
      
      if 1%length(xytsp.t(idsp))>4
        npass=npass+1;
        
        ids=[nstep:nstep+ie-2];
        passes(npass).ids=ids;
        passes(npass).x=xyt.x(ids);
        passes(npass).y=xyt.y(ids);
        passes(npass).t=xyt.t(ids);
          
        dx=passes(npass).x(2:end)-passes(npass).x(1:end-1);
        dy=passes(npass).y(2:end)-passes(npass).y(1:end-1);
        dl=sqrt(dx.^2 + dy.^2);
        
        passlength=[0 cumsum(dl')];
        cutx=passes(npass).x(end)-passes(npass).x(1);
        cuty=passes(npass).y(end)-passes(npass).y(1);
        passes(npass).tortuosity=passlength(end)/sqrt((cutx^2+cuty^2));

        dt=passes(npass).t(2:end)-passes(npass).t(1:end-1);
        rspeed=dl./dt;
        idzero=find(dt==0);
        rspeed(idzero)=nan;
        passes(npass).v(1)=rspeed(1);
        passes(npass).v(2:length(rspeed))=(rspeed(1:end-1)+rspeed(2:end))/2;
        passes(npass).v(end+1)=rspeed(end);
        
        % Smoothening, Sampling rate = 31.5 Hz
%         dx=conv(dx,[1 1 1 1 1 1 0],'same')/6;
%         dy=conv(dy,[1 1 1 1 1 1 0],'same')/6;
        %
        passes(npass).angle=angle(dx+dy*i);
        passes(npass).angle(end+1)=passes(npass).angle(end);
        for nsp=1:length(idsp)
          spiketime=xytsp.t(idsp(nsp));
          j0=max(find(passes(npass).t-spiketime<0));
          tt0=passes(npass).t(j0);
          a0=passes(npass).angle(j0);
          v0=passes(npass).v(j0);
          x0=passes(npass).x(j0);
          y0=passes(npass).y(j0);
          l0=passlength(j0);
          if j0==length(passes(npass).t) | length(j0)==0
            continue;
          end
          tt1=passes(npass).t(j0+1);
          a1=passes(npass).angle(j0+1);
          v1=passes(npass).v(j0+1);
          x1=passes(npass).x(j0+1);
          y1=passes(npass).y(j0+1);
          l1=passlength(j0+1);
      
          z=exp(i*a0) + (exp(i*a1)-exp(i*a0))*(spiketime-tt0)/(tt1-tt0);%linear interpolation
          passes(npass).spikeangle(nsp)=angle(z);
          passes(npass).spikev(nsp)=v0+(v1-v0)/(tt1-tt0)*(spiketime-tt0);
          passes(npass).spikex(nsp)=x0+(x1-x0)/(tt1-tt0)*(spiketime-tt0);
          passes(npass).spikey(nsp)=y0+(y1-y0)/(tt1-tt0)*(spiketime-tt0);
          passes(npass).ell(nsp)=(l0+(l1-l0)/(tt1-tt0)*(spiketime-tt0))/ ...
              passlength(end);

          [dum idmx]=min(abs(field.pf.X(1,:)-passes(npass).spikex(nsp)));
          [dum idmy]=min(abs(field.pf.Y(:,1)-passes(npass).spikey(nsp)));
          
          vvec=[dx'; dy'];
          r=[passes(npass).spikex(nsp) passes(npass).spikey(nsp)]-[xc yc];
          heading = sign(r*vvec(:,j0));
          
          passes(npass).drz(nsp)=drzmap(idmy,idmx)*heading;
        
        end
        passes(npass).tsp=xytsp.t(idsp);
      
      end
    end
    
    nstep=nstep+ie;
    
  end