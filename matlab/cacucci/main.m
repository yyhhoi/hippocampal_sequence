%% Example of how i call mlm for simulated place field data
clear

%% Creating artificial data with directionality
duration=30*60; %seconds
dt=.005; %5 ms
[pos tax]=freeforage(duration,dt);
hd=heading(pos);

%% Generate spike data
rseed=183;
[spiketimes spikepos spikehd]=pffire(tax,pos',hd',rseed);



%%
%spiketimes is 1 x totspks matrix of spiketimes 
%spikehd is 1 x totspks matrix of headdirections during spikes
%spikepos is 2 x totspks matrix of x,y position during spikes
%

[xbin ybin rpos hdbin rhd] = mlm(pos', hd', spikepos, spikehd);


figure(1);
clf
subplot(2,2,1)
imagesc(xbin, ybin, rpos')
set(gca,'YDir','normal')
axis square
subplot(2,2,2)
polar(2*pi*hdbin,rhd);

%simple directionalilty estimate
(max(rhd)-min(rhd))/(max(rhd)+min(rhd))

%% save file for testing

data.pos = pos;
data.tax = tax;
data.hd = hd;
data.spiketimes = spiketimes;
data.spikepos = spikepos;
data.spikehd = spikehd;
save('data.mat','data','-v7')