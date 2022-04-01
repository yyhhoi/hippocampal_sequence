function [fields, pf]=getfields(idat,xytsp)
  
  xyt.x=idat.x;
  xyt.y=idat.y;
  xyt.t=idat.t;
  
  pf = placefield(xyt,xytsp);
  normmap=pf.map/max(max(pf.map));
  [xdim, ydim]=size(normmap);
  
  clines=contourc(pf.X(1,:),pf.Y(:,1),normmap,[0:.1:1]);
  
  
  fields=[];
  ncol=0;
  nf=0;
  while ncol<size(clines,2)
    
    val=clines(1,ncol+1);
    dim=clines(2,ncol+1);
    
    if val==0.2

      xy=clines(:,ncol+2:ncol+1+dim);
      x0=xy(1,1);y0=xy(2,1);x1=xy(1,end);y1=xy(2,end);
      if x0-x1 | y0-y1
        if x0==x1 | y0==y1
          xy(:,end+1)=xy(:,1);
        else
          if x0==pf.X(1,1) | x0==pf.X(1,end)
            xy(1,end+1)=x0;
            xy(2,end)=y1;
            if prod(xy(:,end) == xy(:,1))==0
              xy(:,end+1)=xy(:,1);
            end
          elseif x1==pf.X(1,1) | x1==pf.X(1,end)
            xy(1,end+1)=x1;
            xy(2,end)=y0;
            if prod(xy(:,end) == xy(:,1))==0
              xy(:,end+1)=xy(:,1);
            end
          elseif y0==pf.Y(1,1) | y0==pf.Y(end,1)
            xy(1,end+1)=x1;
            xy(2,end)=y0;
            if prod(xy(:,end) == xy(:,1))==0
              xy(:,end+1)=xy(:,1);
            end
          else
            xy(1,end+1)=x0;
            xy(2,end)=y1;
            if prod(xy(:,end) == xy(:,1))==0
              xy(:,end+1)=xy(:,1);
            end
          end
        end
      end
      
      %
      mask=poly2mask(xy(1,:)-pf.X(1,1),xy(2,:)-pf.Y(1,1),xdim,ydim);
      
      if max(max(mask.*pf.rates))>1 && sum(sum(mask)) > 5^2
        nf=nf+1;
        
        fields(nf).xyval=xy;
        fields(nf).mask=mask;
        fields(nf).pf=pf;
        fields(nf).peakrate=max(max(mask.*pf.rates));
        fields(nf).area= sum(sum(mask));
      end
    
    end
    ncol=ncol+dim+1;
    
  end
