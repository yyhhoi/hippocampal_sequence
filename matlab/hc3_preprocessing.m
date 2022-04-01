
cd('/home/yiu/projects/hippocampus/matlab');
addpath('/home/yiu/projects/hippocampus/matlab/lib/crcns-hc2-scripts');

data_dir = '/home/yiu/projects/hippocampus/data/crcns-hc-3';
topdirs = [dir(fullfile(data_dir, 'ec*')); dir(fullfile(data_dir, 'i01*'))];
seed = 0;
for ndir=1:length(topdirs)
    
    topdir = fullfile(data_dir, topdirs(ndir).name);
    
    sessdirs = dir(topdir);
    sessdirs(1:2) = [];
    
    for nsess=1:length(sessdirs)
        
        sessname = sessdirs(nsess).name;
        sessdir = fullfile(topdir, sessname);
        
        sessxml = LoadPar(fullfile(sessdir, [sessname '.xml']));
        nChan = sessxml.nChannels;
        s = RandStream('mt19937ar', 'Seed', seed);
        nChan_small = randsample(s, nChan-1, 30)';
        eegpth = fullfile(sessdir, [sessname '.eeg']);
        sesslfp = LoadBinary(eegpth, nChan_small+1);
        
        matpth = fullfile(sessdir, [sessname '.mat']);
        fprintf('Saveing\n%s\nto\n%s\n', eegpth, matpth);
        save(matpth, 'sesslfp', '-v7');
        seed = seed + 1;
    end
    
end