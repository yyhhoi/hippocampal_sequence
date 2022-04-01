function out=tuning(theta,nbins)
  
  loglike = @(phi,k,pb) cos(phi-pb)*k - log(2*pi*besseli(0,k));
  
  karr=[0 10.^(-2:.0125:2)];
  garr=besseli(1,karr)./besseli(0,karr);
  
  %% MLfit
  z=nanmean(exp(i*theta)); % mean heading in complex exp
      
  out.pb=angle(z); % mean heading in rad
  RayleighR=abs(z); % resultant length
  
  if sum(~isnan(theta))>0
    ii0=max(find(garr<RayleighR));
    ii1=ii0+1;
    if ii1>length(garr)
      out.k=karr(end);
    else
      k0=karr(ii0);k1=karr(ii1);g0=garr(ii0);g1=garr(ii1);
      out.k=k0 + (k1-k0)/(g1-g0)*(RayleighR-g0);
    end
    out.ll=nanmean(loglike(theta,out.k,out.pb));
  else
    out.k=nan;
    out.ll=nan;
  end
    %%
  
  dbin=2*pi/nbins;

  d0arr=0:dbin:2*pi;
  
  out.darr=mod(d0arr(1:end-1)+out.pb-pi/nbins,2*pi);
  
  idpi=find(out.darr>pi);
  out.darr(idpi) = out.darr(idpi)-2*pi;