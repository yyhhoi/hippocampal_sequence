function out=correctprior(in,occ)
  
  a=in.k*exp(-i*in.pb)-occ.k*exp(-i*occ.pb);
  out.k=abs(a);
  out.pb=angle(conj(a));
  
  nbins=length(in.darr);
  
  dbin=2*pi/nbins;

  d0arr=0:dbin:2*pi;
  out.darr=mod(d0arr(1:end-1)+out.pb-pi/nbins,2*pi);
  
  idpi=find(out.darr>pi);
  out.darr(idpi) = out.darr(idpi)-2*pi;
  out.ll=in.ll; 