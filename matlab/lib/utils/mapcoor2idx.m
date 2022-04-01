function [ idx, idy ] = mapcoor2idx( x, y, xmesh, ymesh )
% Convert map idx to coordinates
% x : One column vector with all entries being x-coordinates
% y : One column vector with all entries being y-coordinates

    xax=xmesh(1,:);
    yax=ymesh(:,1)';
    dmatx=((x*ones(size(xax))-ones(size(x))*xax)).^2;
    dmaty=((y*ones(size(yax))-ones(size(y))*yax)).^2;
    [dum idx]=min(dmatx');
    [dum idy]=min(dmaty');

end

