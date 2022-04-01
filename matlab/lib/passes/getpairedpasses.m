function passes = getpairedpasses(field1, field2, xyt, xytsp1, xytsp2)
% GETPAIREDPASSES get the passes that cross the union of field 1 and 2
%   passes = GETPAIREDPASSES(field1, field2, xyt, xytsp1, xytsp2)
%   field1: structured data of ONE place field. See getfields() function,
%   which gives output fields, of which the fields(i) will be the argument.
%   field2: see field1. Another field that you want to pair with field1.
%   xyt: structured data with fields of x, y, t. All with (N, 1), N =
%   samples of positions.
%   xytsp1: structured data with fields of x, y, t. All with (M, 1), M =
%   samples of spikes. It is the spike information from field1
%   xytsp2: see xytsp1. Spike information from field2.

    minpasslength= 5*.1;% s
  
    mask_union = field1.mask + field2.mask;
    mask = 1.*mask_union;
    mask(mask_union==0)=nan;
    
    [idx, idy] = mapcoor2idx(xyt.x, xyt.y, field1.pf.X, field1.pf.Y); 
    
    passes=[];
    for nstep=1:length(idy)
        tok(nstep)=mask(idy(nstep),idx(nstep));
    end
    nstep=1;
    npass=0;
    while nstep<length(tok)

        trest=tok(nstep:end);
        
        % Set the first step to those within the masked map
        if isnan(trest(1))
            i0=min(find(~isnan(trest)));
            if length(i0)==0
                break;
            end
            nstep=nstep+i0-1;
            trest=tok(nstep:end);
        end
        
        % Set the final step: either it exits the map or the last step within the map
        ie=min(find(isnan(trest)));
        if length(ie)==0
            ie=length(trest)+1;
        end

        % Retrieve the times, corresponding to the steps, from the 'indata'
        tend=xyt.t(nstep+ie-2);
        tstart=xyt.t(nstep);
    
        if tend-tstart>minpasslength % the pass length must last for at least some time
            [passes, npass]=buildpasses(passes, xyt, xytsp1, xytsp2, tstart, tend, npass, nstep, ie);
        end
    
        nstep=nstep+ie;
    end
end