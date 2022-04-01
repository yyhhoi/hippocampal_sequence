function pairout = pair_criterion(fieldA, fieldB)
  
    minspikepairs = 16;
    pairout.ok = 0;
    passA=straightrank(fieldA.passes);
    passB=straightrank(fieldB.passes);
    stimesA=[];    
    
    for n=1:length(passA)
        stimesA=[stimesA passA(n).tsp'];
    end

    stimesB=[];
    for n=1:length(passB)
        stimesB=[stimesB passB(n).tsp'];
    end
    
    num_spikepairs = 0;
    if length(stimesA)*length(stimesB)==0
        isi=[];
    else
        for A_idx=1:length(stimesA) 
            isi_each = stimesA(A_idx) - stimesB;
            id0=find(abs(isi_each(1:end))<.15);
            num_spikepairs = num_spikepairs + length(id0);
            if num_spikepairs > minspikepairs-1
                break;
            end
        end
        
%         isi = stimesA'*ones(size(stimesB)) - ones(size(stimesA'))*stimesB;
    end
    
%     id0=find(abs(isi(1:end))<.15);
%     spikepairs=length(id0);
%     num_spikepairs=sum(spikepairs);
    
    if num_spikepairs > minspikepairs-1
        pairout.ok = 1;
    end
end
  