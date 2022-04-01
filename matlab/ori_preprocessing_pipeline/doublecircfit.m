function  [phi0, phi1, theta0, ost]=doublecircfit(thetaC,phiC)
  
%
plotflag=0;     
     
%
  tarr=-pi:.15:pi;
  sigy=pi/5;
  sigx=pi/5;
  dnsty0=zeros(length(tarr));
  ovec=ones(size(tarr));
  gauss = @(x) exp(cos(tarr'-x(2))*ovec/sigy^2 + ovec'*cos(tarr-x(1))/sigx^2 ...
                   - 2/(sigx^2+sigy^2))/2/pi/sigx/sigy;
  
  for npnt=1:length(phiC)
    dnsty0=dnsty0+gauss([thetaC(npnt) phiC(npnt)]); 
    
  end
  %h0=hist(thetaC,tarr);
  dnsty=dnsty0./(ovec'*sum(dnsty0));
  [~,ism]=max(max(dnsty));
  [~,ismm] = max(dnsty(:,ism));
  

  
  phi1=.0;
  theta0=tarr(ism);
  eta=1;
  
  zmat=dnsty.*exp(i*(tarr'*ovec-phi1*ovec'*cos(tarr-theta0)));    
  z=zmat(1:end);
  
  ttmp=repmat(tarr',1,length(tarr))';
  thetalabels=ttmp(1:end);
  oovec=ones(size(thetalabels));
  q=exp(i*(oovec'*thetalabels)) - exp(i*(thetalabels'*oovec));
  
  F=sum(sum((z')*z.*q));
    
  err=abs(F);
  %imag(F)
  ncnt=0;
  while err > 1e-8*length(thetalabels)
    err0=err;
    
    cs=cos(thetalabels-theta0);
    sn=sin(thetalabels-theta0);
    d1F=i*sum(sum((z')*z.*q.*(cs'*ones(size(cs)) - ones(size(cs'))*cs)));
    d2F=i*phi1*sum(sum((z')*z.*q.*(sn'*ones(size(cs)) - ones(size(cs'))*sn)));
    
    
    M=[real(d1F) real(d2F); imag(d1F)  imag(d2F)];
    x=[real(F); imag(F)];
    dl= -inv(M'*M+ones(2)*.001)*M'*x;
    %dl= -inv(M)*x;

    phi1   = phi1  + dl(1)*eta;
    theta0 = theta0+ dl(2)*eta;
    
    theta0=mod(theta0,2*pi);
    if abs(phi1)>pi
      phi1=0;
    end
      
    zmat=dnsty.*exp(i*(tarr'*ovec-phi1*ovec'*cos(tarr-theta0)));    
    z=zmat(1:end);
  
    F=sum(sum((z')*z.*q));
    
    
    
    err=abs(F);
    ncnt = ncnt+1;
    lambda(ncnt).err=err;
    lambda(ncnt).theta0=theta0;
    lambda(ncnt).phi1=phi1;
    phi0=angle(sum(z));  
    lambda(ncnt).L=mean(mean(dnsty.*cos(tarr'*ovec-phi0-phi1*ovec'* ...
                                        cos(tarr-theta0))));
    
     %err
    
    if ncnt>500
      if eta >5e-4
        ncnt=1;
        [~,idm] = max([lambda(:).L]);
        theta0=lambda(idm).theta0;
        phi1=lambda(idm).phi1;
        eta=eta/3;
        
        
        zmat=dnsty.*exp(i*(tarr'*ovec-phi1*ovec'*cos(tarr-theta0)));    
        z=zmat(1:end);
        phi0=angle(sum(z));  
        F=sum(sum((z')*z.*q));
        err=abs(F);
        lambda(ncnt).err=err;
        lambda(ncnt).theta0=theta0;
        lambda(ncnt).phi1=phi1;
        
        lambda(ncnt).L=mean(mean(dnsty.*cos(tarr'*ovec-phi0-phi1*ovec'* ...
                                        cos(tarr-theta0))));
        [err eta]
      else
        [~,idm] = min([lambda(:).err]);
        theta0=lambda(idm).theta0;
        phi1=lambda(idm).phi1;
        zmat=dnsty.*exp(i*(tarr'*ovec-phi1*ovec'*cos(tarr-theta0)));    
        z=zmat(1:end);
  
        F=sum(sum((z')*z.*q));
        err=abs(F);
        fprintf('Doublecircfit terminated prematurely\n')
        break
      end
    end
  end
  
  if phi1<0
    phi1=abs(phi1);
    theta0 = mod(theta0 + pi, 2*pi);
  end
  if theta0>pi
    theta0=theta0-2*pi;
  end
  
  phi0=angle(sum(z));  
  %err
 
  if plotflag
    figure;
    
    hold on
    imagesc(tarr,[tarr-2*pi tarr tarr+2*pi ]',[dnsty; dnsty; dnsty])
    plot(thetaC,phiC,'.k')
    plot(thetaC,phiC+2*pi,'.k')
    plot(thetaC,phiC-2*pi,'.k')
    plot(tarr,phi0+phi1*cos(tarr-theta0), '-r')
    hold off
    axis square
    set(gca,'Xlim',[-pi pi])
    set(gca,'Ylim',[-3*pi 3*pi])
    xlabel('Heading (rad)')
    ylabel('\theta phase (rad)')
  end
  
  
  ost.err=err; % Optimization error
  ost.L=mean(mean(dnsty.*cos(tarr'*ovec-phi0-phi1*ovec'*cos(tarr- ...
                                                    theta0))));  % Loss objective
  ost.dnsty=dnsty; % normalized heatmap of the probability of observations
  ost.tarr=tarr;  % time array
  %ost.bias=sum(dnsty0);