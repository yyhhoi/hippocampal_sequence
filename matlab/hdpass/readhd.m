clear

pth='/home/leibold/projects/EMankin Data/Principal Cells & Path/';
lfppth='/home/leibold/projects/EMankin Data/LFP/';
%

rats=dir(strcat([pth 'rat*']));
thresharr=[-10 0 1 3 5 7];
nstot=0;
for nrat=1:length(rats)
  
  sessions=dir(strcat([pth rats(nrat).name '/20*']));
  
  for nsess=1:length(sessions)
    [nrat nsess]
    if nrat==7 && nsess==7
      continue; %broken LFP file
    end
    
    nstot=nstot+1;
    
    folder=strcat([pth rats(nrat).name '/'  sessions(nsess).name ...
                     '/Analysis/ProcessedData']);
      
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
          %xytsp.a=spikeData.angleSp{ntrial,nu};
          %xytsp.v=spikeData.vSp{ntrial,nu};
          fields =getfields(indata(ntrial),xytsp);
          for nf=1:length(fields)
            passes=getpasses(fields(nf),indata(ntrial),xytsp,[]);
            hdinfo=gethd(fields(nf),indata(ntrial),xytsp);
            
            for mt=1:length(thresharr)
              hdinfo.passes(mt)=gethdpasses(passes,thresharr(mt));
            end
            
            fieldsintrial=[fieldsintrial hdinfo];
          end
          
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

    day(nstot).trials=trials;
    day(nstot).animal=rats(nrat).name;
    day(nstot).date=sessions(nsess).name;
    clear trials

  end

  
  
end


% save('hd.mat','day','-v7.3')
