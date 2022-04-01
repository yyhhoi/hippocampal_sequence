function [rxbin rybin positionality rhdbin directionality] = mlm(pos, hd,  spikepos,  spikehd);
  
  
  %spikepos (2 x totspks) matrix of positions at spike
  %hd 1 x totspks  matrix of head directions at spike 
    
  %(unit of hd: -.5 to +.5 cycles, 0 cycles points in x-axis direction)
  
  %position
  % pos: (2 x T) matrix 
  %
  %space bin
  spbin=0.05; %5cm bin
  
  % head direction
  % hd 1x T Matrix (in units of cyles -.5 to .5 cycles)
  %
  %angular bin
  abin=1/36; % 0.0278 cycles = 10 degrees
  
  %scaling factor for peak rate estimates (dt for position data)
  dt=1;
  
  %get box size
  maxx=max(pos(1,:))+spbin;
  maxy=max(pos(2,:))+spbin;
  minx=min(pos(1,:))-spbin;
  miny=min(pos(2,:))-spbin;
  
  %boundaries of position and head direction bins
  xbin=[minx:spbin:maxx];
  ybin=[miny:spbin:maxy];
  hdbin=-.5:abin:.5;
  %
  
  %compute n and t from data
  for nx=2:length(xbin)
    [nx-1]
    x0=xbin(nx-1); x1=xbin(nx);
    l1tmp=(pos(1,:)>=x0).*(pos(1,:)<x1);
    l1stmp=(spikepos(1,:)>=x0).*(spikepos(1,:)<x1);
    
    for ny=2:length(ybin)
      y0=ybin(ny-1); y1=ybin(ny);
      l2tmp=(pos(2,:)>=y0).*(pos(2,:)<y1);
      l2stmp=(spikepos(2,:)>=y0).*(spikepos(2,:)<y1);

      for na=2:length(hdbin);
        a0=hdbin(na-1); a1=hdbin(na);
               
        ltmp=l1tmp.*l2tmp.*(hd>=a0).*(hd<a1);
        
        tocc(nx-1,ny-1,na-1)= length(find(ltmp))*dt;
               
        ltmp=l1stmp.*l2stmp.*(spikehd>=a0).*(spikehd<a1);
        
        nspk(nx-1,ny-1,na-1) = length(find(ltmp));
       
      end
    end
  end
  totspks=sum(sum(sum(nspk)));
  %%
 
   
  %% iterate mlm model
  directionality=ones(1,length(hdbin)-1)/(length(hdbin)-1)*sqrt(totspks);
  positionality=ones(length(xbin)-1,length(ybin)-1)/(length(xbin)-1)/(length(ybin)-1)*sqrt(totspks);
  
  err=2;
  while err>.01
      
    % pi = < nij >_j / < dj * tij >_j
    tmp=zeros(size(positionality));
    for na=1:length(directionality)
      tmp=tmp+directionality(na)*tocc(:,:,na);
    end
    ptmp=sum(nspk,3)./tmp;
    
    % dj = < nij >_i / < pi * tij >_i
    tmp=zeros(size(directionality));
    for nx=1:size(positionality,1)
      for ny=1:size(positionality,2)
        tmp=nansum([tmp; positionality(nx,ny)*squeeze(tocc(nx,ny,:))']);
      end
    end
    dtmp=squeeze(sum(sum(nspk,1),2))'./tmp;
    
    
    % nfac means normalization factor, = sum_ij{pi*dj*tij}
    tmp=zeros(size(positionality));
    for na=1:length(directionality)
      tmp=tmp + dtmp(na)*tocc(:,:,na);
    end
    nfac=nansum(nansum(tmp.*ptmp));
    dtmp=dtmp*sqrt(totspks/nfac);
    ptmp=ptmp*sqrt(totspks/nfac);
    
    errd=nanmean(directionality-dtmp)^2
    errp=nanmean(nanmean(positionality-ptmp))^2
    err=sqrt(errd+errp);
    
    
    
    positionality=ptmp;
    directionality=dtmp;
  end
  
  
  %redo axis to indicate bin centers instead of boundaries
  rxbin=xbin(1:end-1)+spbin/2;
  rybin=ybin(1:end-1)+spbin/2;
  rhdbin=hdbin(1:end-1)+abin/2;