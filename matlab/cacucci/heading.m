function hd=heading(pos);
  
  dpos=pos(2:end,:)-pos(1:end-1,:);
  
  hd=angle(dpos(:,1) + i*dpos(:,2))/2/pi;
  hd(end+1)=hd(end);