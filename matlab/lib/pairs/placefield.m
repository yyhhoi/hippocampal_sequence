function pf = placefield(occupancy, X, Y, xytsp)
% PLACEFIELD calculate the map of firing
%   occupancy: 2D occupancy map of the rat. Obtained from getoccupancy()
%   function.
%   X, Y: meshgrids of the environment. Obtained from getoccupancy()
%   function.
%   xytsp: structured data with fields of x, y, t. All with (M, 1), M =
%   samples of spikes.
%   pf: structured data with fields of X (x-meshgrid), Y (y-meshgrid), 
%       occ (2D occupanccy map), rates (2D rate map of the field), map (rates/occ)
  
  sig= 3.;% in cm
  gauss = @(x,y) exp(-(x.^2+y.^2)/(2*sig^2))/(2*pi*sig^2);
  
  
  rates=zeros(size(X));
  for nt=1:length(xytsp.t)
    rates=rates+gauss(xytsp.x(nt)-X,xytsp.y(nt)-Y);
  end
  
  
  pf.map=rates./(0.0001+occupancy);
  pf.X=X;
  pf.Y=Y;
  pf.occ=occupancy/sum(sum(occupancy));
  pf.rates=rates;
  