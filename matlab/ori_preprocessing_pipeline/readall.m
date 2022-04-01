clear

pth='/home/leibold/projects/EMankin Data/Principal Cells & Path/';
lfppth='/home/leibold/projects/EMankin Data/LFP/';
%

rats=dir(strcat([pth 'rat*']));

nstot=0;
for nrat=1:length(rats)
  
  sessions=dir(strcat([pth rats(nrat).name '/20*']));
  
  for nsess=1:length(sessions)
    [nrat nsess]
    if nrat==7 && nsess==7
      continue; %broken LFP file
    end
    
    nstot=nstot+1;
    
    folder=strcat([pth rats(nrat).name '/'  sessions(nsess).name '/Analysis/ProcessedData']);
    LFPfolder=strcat([lfppth rats(nrat).name '/'  sessions(nsess).name '/Analysis/ProcessedData']);
    
    load(strcat([LFPfolder '/rawLFP__Full.mat']));
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
      dum
      wave(ntrial).theta=filtfilt(B, A, rawLFPData(ntrial,idm).sample);
      wave(ntrial).lfp=rawLFPData(ntrial,idm).sample;
      wave(ntrial).tax=rawLFPData(ntrial,idm).timestamps;
      wave(ntrial).phase=angle(hilbert(wave(ntrial).theta));
    end
    clear rawLFPData;

    
    for castr={'CA1', 'CA2', 'CA3'}

      tstr=strcat([castr{1} '.mat']);
      dtest=dir(strcat([folder '/cellInfo*' tstr]));
      if length(dtest)==0
        continue
      end
      suffix=dtest.name(9:end)
      
      load(strcat([folder '/cellInfo' suffix]));
      load(strcat([folder '/ctxID' suffix]));
      load(strcat([folder '/indata' suffix]));
      load(strcat([folder '/spikeData' suffix]));
      
      nunits=size(spikeData.tSp,2);
      for ntrial=1:length(indata)
        
        
        fieldsintrial=[];
        for nu=1:nunits
          xytsp.x=spikeData.xSp{ntrial,nu};
          xytsp.y=spikeData.ySp{ntrial,nu};
          xytsp.t=spikeData.tSp{ntrial,nu};
          xytsp.a=spikeData.angleSp{ntrial,nu};
          xytsp.v=spikeData.vSp{ntrial,nu};
          fields =getfields(indata(ntrial),xytsp);
          for nf=1:length(fields)
            fields(nf).passes=getpasses(fields(nf),indata(ntrial),xytsp, wave(ntrial));

          end
          nft=length(fieldsintrial);
          fieldsintrial= [fieldsintrial fields];
        end
        
        if str2num(castr{1}(3))==1%CA1
          trials(ntrial).CA1fields=fieldsintrial;
          trials(ntrial).nfields(1)= length(fieldsintrial)
        elseif str2num(castr{1}(3))==2%CA2
          trials(ntrial).CA2fields=fieldsintrial;
          trials(ntrial).nfields(2)= length(fieldsintrial)
        elseif str2num(castr{1}(3))==3%CA3
          trials(ntrial).CA3fields=fieldsintrial;
          trials(ntrial).nfields(3)= length(fieldsintrial)
        end
        trials(ntrial).shape=ctxID{ntrial};

      end
    end
    day(nstot).lfp=wave;
    day(nstot).trials=trials;
    day(nstot).animal=rats(nrat).name;
    day(nstot).date=sessions(nsess).name;
    clear trials
    clear wave
  end

  
  
end


save('days.mat','day','-v7.3')
