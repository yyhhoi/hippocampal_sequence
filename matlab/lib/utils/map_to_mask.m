function [ infield] = map_to_mask(x, y, xmesh, ymesh, mask)


    [idx, idy] = mapcoor2idx(x, y, xmesh, ymesh);
    
    for nstep=1:length(idy)
        infield(nstep)=mask(idy(nstep),idx(nstep));
    end
        
end

