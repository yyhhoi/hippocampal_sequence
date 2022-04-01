function [ info_persecond, info_perspike ] = spatial_information( mean_rate_loc, prob_loc )
%SPATIAL_INFORMATION Spatial information content
%   Detailed explanation goes here

    marginal_rate = sum(sum((mean_rate_loc .* prob_loc)));
    
    integrand = mean_rate_loc .* log2(mean_rate_loc ./ marginal_rate) .* prob_loc;
    
    info_persecond = sum(sum(integrand));
    info_perspike = info_persecond/marginal_rate; 

end

