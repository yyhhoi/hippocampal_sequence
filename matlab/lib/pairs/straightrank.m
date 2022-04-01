function passout = straightrank(passin)
% Original Method
speedlimit = 3;
thresh = 5;

passout=passin;

for np=1:length(passin)
  
  
  meanspeed(np)=mean(passin(np).v);
  meanangle(np)=angle(sum(exp(i*passin(np).angle')));
  meanR(np)=abs(mean(exp(i*passin(np).angle')));
  
  nspikes=length(passin(np).tsp);
  
 
  ntmp=length(passin(np).angle);
 
  passout(np).straightangle =  meanangle(np);
  passout(np).nstats =  [ntmp nspikes];
  if nspikes>3 & meanspeed(np)>speedlimit & meanR(np)^2 > (1+thresh*sqrt(1-1/ntmp))/ntmp
    passout(np).straightR = meanR(np);
  else
    passout(np).straightR = nan;
  end
  
  
end

