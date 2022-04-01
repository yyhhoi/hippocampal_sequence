% All-included piplines
clear;



pth='/home/yiu/projects/hippocampus/data/emankin/principal_cells_and_path/';
lfppth='/home/leibold/projects/EMankin Data/LFP/';
addpath('/home/yiu/projects/hippocampus/matlab/lib');

rats=dir(fullfile(pth, 'rat*'));
nstot=0;
for nrat=1:length(rats)

    sessions=dir(fullfile(pth, rats(nrat).name, '20*'));

    for nsess=1:length(sessions)
        disp(sprintf('%d/%d rat, %d/%d session', nrat, length(rats), nsess, length(sessions)));

        if nrat==7 && nsess==7
            continue; %broken LFP file
        end

        
        folder=fullfile(pth, rats(nrat).name, sessions(nsess).name, ...
            'Analysis', 'ProcessedData');
        LFPfolder=fullfile(lfppth, rats(nrat).name, sessions(nsess).name, ...
            'Analysis', 'ProcessedData');
        
        % LFP processing
        load(fullfile(LFPfolder, 'rawLFP__Full.mat'));
        dt=1/(rawLFPData(1,1).Fs(1));
        thetaband=[5 12]*dt*2;
        deltaband=[1 4]*dt*2;
        [B,A] = butter(4,thetaband);
        [Bd,Ad] = butter(2,deltaband);
        for ntrial=1:size(rawLFPData,1)
            trials(ntrial).nfields=[0 0 0];
            for nchannel=1:size(rawLFPData,2)
                theta = filtfilt(B, A, rawLFPData(ntrial,nchannel).sample);
                delta = filtfilt(Bd, Ad, rawLFPData(ntrial,nchannel).sample);
                thetadeltaratio(nchannel)=sum(theta.^2)/sum(delta.^2);
            end
            [dum idm]=max(thetadeltaratio(nchannel));
            idm
            wave(ntrial).theta=filtfilt(B, A, rawLFPData(ntrial,idm).sample);
            wave(ntrial).lfp=rawLFPData(ntrial,idm).sample;
            wave(ntrial).tax=rawLFPData(ntrial,idm).timestamps;
            wave(ntrial).phase=angle(hilbert(wave(ntrial).theta));
        end
        clear rawLFPData;
        
        % Spike & Indata processing
        for CA_name={'CA1', 'CA2', 'CA3'}
            CA_name = char(CA_name);
            
            tstr=strcat([CA_name '.mat']);  % sth like 'CA1.mat'
            dtest=dir(fullfile(folder, strcat(['cellInfo*' tstr])));
            
            
            if length(dtest)==0
                continue
            end
            suffix=dtest.name(9:end); % sth like '_Day1_CA1.mat'
            
            load(fullfile(folder, strcat(['cellInfo' suffix])));
            load(fullfile(folder, strcat(['ctxID' suffix])));
            load(fullfile(folder, strcat(['indata' suffix])));
            load(fullfile(folder, strcat(['spikeData' suffix])));
            
            nunits=size(spikeData.tSp,2);
            num_trials = length(indata);
            
            for ntrial=1:num_trials  % For each TRIAL
                fieldsintrial= [];
                xyt.x = indata(ntrial).x;
                xyt.y = indata(ntrial).y;
                xyt.t = indata(ntrial).t;
                [occupancy_map, Xmesh, Ymesh] = getoccupancy(xyt);
                for nu=1:nunits  % For each UNIT
                    xytsp = loadspikes(spikeData, ntrial, nu);
                    fields =getfields(occupancy_map, Xmesh, Ymesh, xytsp, 25);
                    for nf=1:length(fields)  % For each FIELD in a UNIT
                        fields(nf).passes=getpasses(fields(nf), xyt, xytsp);
                        fields(nf).nunits = nu;
                        fields(nf).xytsp.xsp = spikeData.xSp{ntrial, nu};
                        fields(nf).xytsp.ysp = spikeData.ySp{ntrial, nu};
                        fields(nf).xytsp.tsp = spikeData.tSp{ntrial, nu};
                    end
                    nft=length(fieldsintrial);
                    fieldsintrial = [fieldsintrial fields];
                end
                
                % Check if each pair of fields should be included
                np=0;
                for nf=1:length(fieldsintrial)  % For all PAIRS of FIELDS between all UNITS
                    for mf=nf+1:length(fieldsintrial)
                        pairtmp=pair_criterion(fieldsintrial(nf),fieldsintrial(mf));
                        pairtmp.fi=[nf mf];
                        if pairtmp.ok  % Suitable pair identified!
                            % Find passes in the paired fields
                            field1 = fieldsintrial(nf);
                            field2 = fieldsintrial(mf);
                            xytsp1 = loadspikes(spikeData, ntrial, field1.nunits);
                            xytsp2 = loadspikes(spikeData, ntrial, field2.nunits);
                            pairedpasses=getpairedpasses(field1, field2, xyt, xytsp1, xytsp2);
                            for npass=1:length(pairedpasses)
                                [infield1, infield2] = run_across_pair(pairedpasses(npass), field1, field2 );
                                pairedpasses(npass).infield1 = infield1;
                                pairedpasses(npass).infield2 = infield2;
                            end
                            pairtmp.pairedpasses = pairedpasses;
                            pairtmp.pairfit = pairanalyze_pairedpasses(pairedpasses);
                            % Store into structured data
                            np = np+1;
                            pair(np)=pairtmp;
                        end
                    end
                end
                
                
                withintrial_idx = nstot + ntrial;

                if str2num(CA_name(3))==1 %CA1
                    day(withintrial_idx).CA1fields=fieldsintrial;
                    if np > 0
                        day(withintrial_idx).CA1pairs=pair;
                    end
                    day(withintrial_idx).nfields(1)= length(fieldsintrial);
                elseif str2num(CA_name(3))==2 %CA2
                    day(withintrial_idx).CA2fields=fieldsintrial;
                    if np > 0
                        day(withintrial_idx).CA2pairs=pair;
                    end
                    day(withintrial_idx).nfields(2)= length(fieldsintrial);
                elseif str2num(CA_name(3))==3 %CA3
                    day(withintrial_idx).CA3fields=fieldsintrial;
                    if np > 0
                        day(withintrial_idx).CA3pairs=pair;
                    end
                    day(withintrial_idx).nfields(3)= length(fieldsintrial);
                end

                clear pair;  % Don't forget to add this
                fprintf('%s: nstot %d, trial %d, idx %d\n', CA_name, nstot, ntrial, withintrial_idx);
                day(withintrial_idx).trials=ntrial;
                day(withintrial_idx).(sprintf('%sindata', CA_name)) = indata(ntrial);
                day(withintrial_idx).wave=wave(ntrial);
                day(withintrial_idx).animal=rats(nrat).name;
                day(withintrial_idx).date=sessions(nsess).name;
                day(withintrial_idx).shape=ctxID{ntrial};
                clear new_xytsp;
            end
        end
        nstot = nstot + num_trials;
        

    end

end

% save('/home/yiu/projects/hippocampus/data/emankindata_processed.mat','day','-v7.3')
