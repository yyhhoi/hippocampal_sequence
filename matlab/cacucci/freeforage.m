function [pcoord tcoord]=freeforage(T,dt)

  %T, dt in seconds
  
boxl=20; % 2m



rEdge=[boxl,boxl];          % The size of the box, arbitrary units
phi=0;                      % Initial Direction to start moving, from 0 to 2*pi
speed= 20/10; % 20 cm/s     % Speed of the Animal, constant
PhiAdvance=pi/36;    


%
r=round([1,1]*boxl/2);        

pcoord=zeros(round(T/dt),2);
tcoord=zeros(round(T/dt),1);

stdtrc=zeros(round(T/dt),1);
mentrc=zeros(round(T/dt),1);

%% Simulation
[X,Y]=meshgrid(0:.5:boxl,0:.5:boxl);

zlow=0;
t_i=1;
time=zeros(round(T/dt),1);


pcoord(1,:)=r;   

t=0;
while t<T
  t_i=t_i + 1;
   t= dt*t_i;
  
  %%%%%%
  phi = phi + PhiAdvance*randn(1);
  r1 = r + speed*dt*[cos(phi),sin(phi)];
  while any(r1>rEdge-.5) || any(r1<[0,0]+.5)
    rc=(r-boxl/2);
    rc=rc/norm(rc);
    phi= angle(exp(i*phi)-(rc(1)+i*rc(2))/boxl);
    r1 = r + speed*dt*[cos(phi),sin(phi)];
  end
  
  r=r1;
  %%%%%%
   
    
  %
  pcoord(t_i,:)=r;
  tcoord(t_i)=t;
   
end

pcoord=pcoord/boxl;