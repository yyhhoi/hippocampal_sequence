function passes = spikepasses( passes, xytsp, passlength, tstart, tend, npass, identifier)

    idsp=find(xytsp.t >= tstart & xytsp.t <= tend); % locate the index of the spikes, by the inclusive time interval
    for nsp=1:length(idsp)
        spiketime=xytsp.t(idsp(nsp));
        j0=max(find(passes(npass).t-spiketime<0));
        if length(j0)==0
            continue
        end
        
        tt0=passes(npass).t(j0);
        a0=passes(npass).angle(j0);
        v0=passes(npass).v(j0);
        x0=passes(npass).x(j0);
        y0=passes(npass).y(j0);
        l0=passlength(j0);
        if j0==length(passes(npass).t) | length(j0)==0
            continue;
        end
        tt1=passes(npass).t(j0+1);
        a1=passes(npass).angle(j0+1);
        v1=passes(npass).v(j0+1);
        x1=passes(npass).x(j0+1);
        y1=passes(npass).y(j0+1);
        l1=passlength(j0+1);

        z=exp(i*a0) + (exp(i*a1)-exp(i*a0))*(spiketime-tt0)/(tt1-tt0);%linear interpolation
        passes(npass).(sprintf('spike%sangle', identifier))(nsp)=angle(z);
        passes(npass).(sprintf('spike%sv', identifier))(nsp)=v0+(v1-v0)/(tt1-tt0)*(spiketime-tt0);
        passes(npass).(sprintf('spike%sx', identifier))(nsp)=x0+(x1-x0)/(tt1-tt0)*(spiketime-tt0);
        passes(npass).(sprintf('spike%sy', identifier))(nsp)=y0+(y1-y0)/(tt1-tt0)*(spiketime-tt0);
        passes(npass).(sprintf('ell%s', identifier))(nsp)=(l0+(l1-l0)/(tt1-tt0)*(spiketime-tt0))/ ...
            passlength(end);

    end
    passes(npass).(sprintf('tsp%s', identifier))=xytsp.t(idsp);
end

