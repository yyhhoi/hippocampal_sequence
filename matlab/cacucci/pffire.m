function [stimes spos shd]=pffire(tax,pos,hd,rseed)

    rng(rseed);
    center=rand(2,1);  % place field centre
    hdc=rand;  % preferred angle
    
    
    
    dt=tax(2)-tax(1);

    rmax=20;  % maximum firing rate
    sig=.1;  % sd for spatial tuning
    shd=.1;  % sd for directional tuning
    a=.6;  % 
    
    % Spatial tuning
    lambda = rmax*exp(-sum( (pos-center*ones(1,size(pos,2))).^2 )/ 2/sig^2);
    % Directional tuning
    lambda = lambda .* (1-a + a * exp((cos( 2*pi*(hd-hdc) )-1)/(2*pi* ...
                                                shd)^2));

    ilam=-log(rand);
    stimes=[];
    spos=[];
    shd=[];
    for nt=1:length(tax)
        % ilam = evolution of the "hazard"
        ilam = ilam - dt*lambda(nt);
        if ilam < 0
            ilam=-log(rand);
            stimes = [stimes nt*dt];
            spos = [spos pos(:,nt)];
            shd = [shd hd(nt)];
        end

    end
end