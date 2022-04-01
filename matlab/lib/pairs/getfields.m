function fields =getfields(occupancy, X, Y, xytsp, min_size)
% GETFIELDS calculate multiple place fields from one neuron
%   occupancy: 2D occupancy map of the rat. Obtained from getoccupancy()
%   function.
%   X, Y: meshgrids of the environment. Obtained from getoccupancy()
%   function.
%   xytsp: structured data with fields of x, y, t. All with (M, 1), M =
%   samples of spikes.
%   fields: structured data with fields 'xyval' (contour of the place
%   field), mask (binary 2D mask of the place field), pf (structure, see
%   placefield() function). This structured data has size W = number of
%   fields found in this neuron. The rate map (pf.rates, pf.map) is
%   multi-modal. The modalities are detected by the algorithm and separated
%   to different masks.

    pf = placefield(occupancy, X, Y, xytsp);
    normmap=pf.map/max(max(pf.map));
    [xdim, ydim]=size(normmap);

    clines=contourc(pf.X(1,:),pf.Y(:,1),normmap,[0:.1:1]);


    fields=[];
    ncol=0;
    nf=0;
    [info_persec, info_persp] = spatial_information(pf.map, pf.occ);
    while ncol<size(clines,2)

        val=clines(1,ncol+1);
        dim=clines(2,ncol+1);

        if val==0.2

            xy=clines(:,ncol+2:ncol+1+dim);
            x0=xy(1,1);y0=xy(2,1);x1=xy(1,end);y1=xy(2,end);
            if x0-x1 | y0-y1  % If either x or y ends differently than the start
                if x0==x1 | y0==y1  % If either x or y ends at the same point as the start
                    xy(:,end+1)=xy(:,1);   % close the polygon
                else  % if both x and y don't end at the start
                    if x0==pf.X(1,1) | x0==pf.X(1,end)  % if x starts at the boundary
                        xy(1,end+1)=x0;
                        xy(2,end)=y1;
                        if prod(xy(:,end) == xy(:,1))==0  % either x/y are different between start and end
                            xy(:,end+1)=xy(:,1);
                        end
                    elseif x1==pf.X(1,1) | x1==pf.X(1,end)  % if x ends at the boundary
                        xy(1,end+1)=x1;
                        xy(2,end)=y0;
                        if prod(xy(:,end) == xy(:,1))==0
                            xy(:,end+1)=xy(:,1);
                        end
                    elseif y0==pf.Y(1,1) | y0==pf.Y(end,1)
                        xy(1,end+1)=x1;
                        xy(2,end)=y0;
                        if prod(xy(:,end) == xy(:,1))==0
                            xy(:,end+1)=xy(:,1);
                        end
                    else
                        xy(1,end+1)=x0;
                        xy(2,end)=y1;
                        if prod(xy(:,end) == xy(:,1))==0
                            xy(:,end+1)=xy(:,1);
                        end
                    end
                end
            end

            %
            mask=poly2mask(xy(1,:)-pf.X(1,1),xy(2,:)-pf.Y(1,1),xdim,ydim);
            aver_rate_inside = mean(mean(pf.map(mask)));
            aver_rate_outside = mean(mean(pf.map(~mask)));

            field_area = sum(sum(mask));

            if (max(max(mask.*pf.rates))>1) && (field_area > min_size) && ...
                (aver_rate_inside > aver_rate_outside)

                nf=nf+1;

                fields(nf).xyval=xy;
                fields(nf).mask=mask;
                fields(nf).pf=pf;
                fields(nf).spatialinfosec= info_persec;
                fields(nf).spatialinfospike = info_persp;
                
                
            end

        end
        ncol=ncol+dim+1;

    end
end
  
 
  