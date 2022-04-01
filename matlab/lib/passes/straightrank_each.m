function [meanangle, nspikes1, nspikes2, straightR1, straightR2] = straightrank_each(pass_each)
%Modified from straightrank.m

    speedlimit = 3;
    thresh = 5;

    meanspeed = mean(pass_each.v);
    meanangle = angle(sum(exp(i*pass_each.angle')));
    meanR = abs(mean(exp(i*pass_each.angle')));
    nspikes1 = length(pass_each.tsp1);
    nspikes2 = length(pass_each.tsp2);
    ntmp = length(pass_each.angle);

    if nspikes1 > 3  & meanspeed>speedlimit & meanR^2 > (1+thresh*sqrt(1-1/ntmp))/ntmp
        straightR1 = meanR;
    else
        straightR1 = nan;
    end

    if nspikes2 > 3  & meanspeed>speedlimit & meanR^2 > (1+thresh*sqrt(1-1/ntmp))/ntmp
        straightR2 = meanR;
    else
        straightR2 = nan;
    end
end

