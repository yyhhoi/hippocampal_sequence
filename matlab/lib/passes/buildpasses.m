function [passes, npass]=buildpasses(passes, xyt, xytsp1, xytsp2, tstart, tend, npass, nstep, ie)

    npass=npass+1;

    ids=[nstep:nstep+ie-2];
    passes(npass).ids=ids;
    passes(npass).x=xyt.x(ids);
    passes(npass).y=xyt.y(ids);
    passes(npass).t=xyt.t(ids);

    dx=passes(npass).x(2:end)-passes(npass).x(1:end-1);
    dy=passes(npass).y(2:end)-passes(npass).y(1:end-1);
    dl=sqrt(dx.^2 + dy.^2);

    passlength=[0 cumsum(dl')];
    cutx=passes(npass).x(end)-passes(npass).x(1);
    cuty=passes(npass).y(end)-passes(npass).y(1);
    passes(npass).tortuosity=passlength(end)/sqrt((cutx^2+cuty^2));

    dt=passes(npass).t(2:end)-passes(npass).t(1:end-1);
    rspeed=dl./dt;
    idzero=find(dt==0);
    rspeed(idzero)=nan;
    passes(npass).v(1)=rspeed(1);
    passes(npass).v(2:length(rspeed))=(rspeed(1:end-1)+rspeed(2:end))/2;
    passes(npass).v(end+1)=rspeed(end);

    % Smoothening, Sampling rate = 31.5 Hz
%     dx=conv(dx,[1 1 1 1 1 1 0],'same')/6;
%     dy=conv(dy,[1 1 1 1 1 1 0],'same')/6;
    %
    passes(npass).angle=angle(dx+dy*i);
    passes(npass).angle(end+1)=passes(npass).angle(end);
    passes = spikepasses( passes, xytsp1, passlength, tstart, tend, npass, '1');
    passes = spikepasses( passes, xytsp2, passlength, tstart, tend, npass, '2');
    
end