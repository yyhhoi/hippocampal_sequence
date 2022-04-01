clear

load('/home/leibold/projects/seqdir/pairs/pairdays.mat')
outf=load('/home/leibold/projects/seqdir/out_t3_.mat');

%%
nftot=[0 0 0];
for nd=1:length(day)
  
  trials=day(nd).trials;
  
  idc=[];ids=[];
  for nt=1:length(trials)

    if trials(nt).shape=='Circle'
      idc=[idc nt]; 
    else
      ids=[ids nt];
    end
    
  end
  
  nMats=reshape([trials(ids).npairs],3,length(ids))';  % num pairs for each ids
  [dums, idmax]=max(nMats);  % square id that has max num of pairs for each CA region
  idbest=ids(idmax);
  if length(dums)==0
    dums=zeros(1,3);
    idbest=nan(1,3);
  end
  
  nMatc=reshape([trials(idc).npairs],3,length(idc))';
  [dumc, idmax]=max(nMatc);
  if length(dumc)==0
    dumc=zeros(1,3);
    idbest(2,:)=nan(1,3);
  else
    idbest=[idbest; idc(idmax)];
  end
  

  dum=[dums; dumc];
  
  for nca=1:3 %CA1, CA2, CA3
    for nshape=1:2
      if dum(nshape,nca)<1
        continue
      end
      
      % This is the line where idbest for each shape is selected
      evstr=sprintf('pairs=trials(idbest(nshape,nca)).CA%dpairs;', nca)     
      eval(evstr);
      
      for nf=1:length(pairs)
        nftot(nca) = nftot(nca)+1
        
        out(nca).res(nftot(nca)).phi1  = pairs(nf).phi1;
        out(nca).res(nftot(nca)).occK=   pairs(nf).occ.k;
        out(nca).res(nftot(nca)).occP=   pairs(nf).occ.pb;
        out(nca).res(nftot(nca)).tuningK= pairs(nf).tuning.k;
        out(nca).res(nftot(nca)).tuningP= pairs(nf).tuning.pb;
        out(nca).info(nftot(nca)).shape=nshape;
        out(nca).info(nftot(nca)).animal=day(nd).animal;
        out(nca).info(nftot(nca)).date=day(nd).date;
        out(nca).info(nftot(nca)).idbest=idbest(nshape,nca);
        info=out(nca).info(nftot(nca));
        ids=find(strcmp(info.date, {outf.out(nca).info.date}) & ...
                 strcmp(info.animal, {outf.out(nca).info.animal}) & ...
                 (info.shape == [outf.out(nca).info.shape]));
        out(nca).info(nftot(nca)).nsfields=ids(pairs(nf).fi);
        
        pbs(1)=outf.out(nca).res(ids(pairs(nf).fi(1))).darr4(1)+pi/6;
        pbs(2)=outf.out(nca).res(ids(pairs(nf).fi(2))).darr4(1)+pi/6;
        
        out(nca).res(nftot(nca)).dpb=angle(exp(i*(pbs(1)-pbs(2))));
        
      end
            
    end
  end
  
  
end

%% Figure 1 a
amp1=mean([out(1).res(:).phi1]);
amp2=mean([out(2).res(:).phi1]);
amp3=mean([out(3).res(:).phi1]);
phi1arr1=hist([out(1).res(:).phi1],0:.1:pi);
phi1arr2=hist([out(2).res(:).phi1],0:.1:pi);
phi1arr3=hist([out(3).res(:).phi1],0:.1:pi);

occKarr1=hist([out(1).res(:).occK],0:.2:7); 
occKarr2=hist([out(2).res(:).occK],0:.2:7);
occKarr3=hist([out(3).res(:).occK],0:.2:7);
tunKarr1=hist([out(1).res(:).tuningK],0:.2:7);
tunKarr2=hist([out(2).res(:).tuningK],0:.2:7);
tunKarr3=hist([out(3).res(:).tuningK],0:.2:7);


nbins=10
dphi=2*pi/nbins;
darr0=(0:nbins)*dphi-pi;

tmp=[out(1).res(:).tuningP];
tunParr1=histc(tmp,darr0);
tunParr1(end) = tunParr1(1);

tmp=[out(2).res(:).tuningP];
tunParr2=hist(tmp,darr0);
tunParr2(end) = tunParr2(1);

tmp=[out(3).res(:).tuningP];
tunParr3=hist(tmp,darr0);
tunParr3(end) = tunParr3(1);
%

darr=[darr0 pi];

pphi13=pstr(ranksum([out(1).res(:).phi1],[out(3).res(:).phi1]),2)
pphi23=pstr(ranksum([out(2).res(:).phi1],[out(3).res(:).phi1]),2)

[~,idmin1]=min(abs(cumsum(phi1arr1)/sum(phi1arr1)-.75)); % Upper quartile?
[~,idmin2]=min(abs(cumsum(phi1arr2)/sum(phi1arr2)-.75));
[~,idmin3]=min(abs(cumsum(phi1arr3)/sum(phi1arr3)-.75));
aax=0:.1:pi;
figure(1);
clf
subplot(3,2,1)
hold on
plot(0:.1:pi,cumsum(phi1arr1)/sum(phi1arr1),'-k')
plot(0:.1:pi,cumsum(phi1arr2)/sum(phi1arr2),'-g')
plot(0:.1:pi,cumsum(phi1arr3)/sum(phi1arr3),'-r')
plot(aax(idmin1)*[1 1], [0 1]*.75, '-k')
plot(aax(idmin2)*[1 1], [0 1]*.75, '-g')
plot(aax(idmin3)*[1 1], [0 1]*.75, '-r')
hold off
set(gca,'Xlim', [0 pi])
set(gca,'XTick',[0:1:3])
xlabel('\phi_1 (rad)')
ylabel('Cdf')
title('All pairs','Fontweight', 'normal')
text(2,.35,strcat(['p_{rs}=' num2str(pphi13)]),'Color','k')
text(2,.15,strcat(['p_{rs}=' num2str(pphi23)]),'Color','g')
%% Figure 1b
doubletun1=[];
doubletun2=[];
doubletun3=[];
doubledarr=[];
for n=1:nbins/2
  doubletun1=[doubletun1 sum(tunParr1([n n+nbins/2]))];
  doubletun2=[doubletun2 sum(tunParr2([n n+nbins/2]))];
  doubletun3=[doubletun3 sum(tunParr3([n n+nbins/2]))];
  
  doubledarr=[doubledarr (n-1)*4*pi/nbins];
end

a1=[out(1).res(:).tuningP]*2;
a2=[out(2).res(:).tuningP]*2;
a3=[out(3).res(:).tuningP]*2;

pr1=circ_rtest(a1(~isnan(a1)))
pr2=circ_rtest(a2(~isnan(a2)))
pr3=circ_rtest(a3(~isnan(a3)))

pv1=circ_vtest(a1(~isnan(a1)),angle(sum(exp(i*a1(~isnan(a1))))))
pv2=circ_vtest(a2(~isnan(a2)),angle(sum(exp(i*a2(~isnan(a2))))))
pv3=circ_vtest(a3(~isnan(a3)),angle(sum(exp(i*a3(~isnan(a3))))))

ctr=0;
for nn=5:5:30
  bin=2*pi/nn;
  N=nn;

  ctr=ctr+1;
  ain=a1(~isnan(a1));
  M=length(ain);
  m=sum(abs(ain)<bin/2);
  pb1(ctr)=binocdf(m,M,1/N)
  ain=a2(~isnan(a2));
  M=length(ain);
  m=sum(abs(ain)<bin/2);
  pb2(ctr)=binocdf(m,M,1/N)
  ain=a3(~isnan(a3));
  M=length(ain);
  m=sum(abs(ain)<bin/2);
  pb3(ctr)=binocdf(m,M,1/N)
end

subplot(3,2,3)
hold on
bar(darr0-.2,(tunParr1)/sum(tunParr1),'k','Barwidth',.3,'Edgecolor','w')
bar(darr0,(tunParr2)/sum(tunParr2),'g','Barwidth',.3,'Edgecolor','w')
bar(darr0+.2,(tunParr3)/sum(tunParr3),'r','Barwidth',.3,'Edgecolor','w')
hold off
set(gca,'Xlim',pi*[-1.2 1.2])
set(gca,'XTick',[-3:3:3])
set(gca,'Ylim',[0 .25]);
xlabel('Rel. heading (rad)')
ylabel('Norm. pair count')

text(angle(sum(exp(i*a1(~isnan(a1)))))/2,.215,num2str(pstr(pv1,2)), ...
     'Color','k')
text(angle(sum(exp(i*a2(~isnan(a2)))))/2,.175,num2str(pstr(pv2,2)), ...
     'Color','g')
text(angle(sum(exp(i*a3(~isnan(a3)))))/2,.175,num2str(pstr(pv3,2)), ...
     'Color','r')

%% figure 1d (not)

id1=find(abs([out(1).res(:).dpb])<pi/3);
id2=find(abs([out(2).res(:).dpb])<pi/3);
id3=find(abs([out(3).res(:).dpb])<pi/3);
tmp1=[out(1).res(:).phi1];
tmp2=[out(2).res(:).phi1];
tmp3=[out(3).res(:).phi1];
phi11=tmp1(id1);
phi12=tmp2(id2);
phi13=tmp3(id3);

phi1arr1=hist(phi11,0:.1:pi);
phi1arr2=hist(phi12,0:.1:pi);
phi1arr3=hist(phi13,0:.1:pi);

pphi13=pstr(ranksum(phi13,phi11),2)
pphi23=pstr(ranksum(phi12,phi13),2)

[~,idmin1]=min(abs(cumsum(phi1arr1)/sum(phi1arr1)-.75));
[~,idmin2]=min(abs(cumsum(phi1arr2)/sum(phi1arr2)-.75));
[~,idmin3]=min(abs(cumsum(phi1arr3)/sum(phi1arr3)-.75));

subplot(3,2,2)
hold on
plot(0:.1:pi,cumsum(phi1arr1)/sum(phi1arr1),'-k')
plot(0:.1:pi,cumsum(phi1arr2)/sum(phi1arr2),'-g')
plot(0:.1:pi,cumsum(phi1arr3)/sum(phi1arr3),'-r')
plot(aax(idmin1)*[1 1], [0 1]*.75, '-k')
plot(aax(idmin2)*[1 1], [0 1]*.75, '-g')
plot(aax(idmin3)*[1 1], [0 1]*.75, '-r')
hold off
set(gca,'Xlim', [0 pi])
set(gca,'XTick',[0:1:3])
xlabel('\phi_1 (rad)')
ylabel('Cdf')
text(1,.35,strcat(['p_{rs}=' num2str(pphi13)]),'Color','k')
text(1,.15,strcat(['p_{rs}=' num2str(pphi23)]),'Color','g')
title('Neighboring pairs', 'Fontweight', 'normal')
%% Figure 1e (not)
nbins=10
dphi=2*pi/nbins;
darr0=(0:nbins)*dphi-pi;

tmp=[out(1).res(:).tuningP];
tmp=tmp(id1);
a1=tmp*2;
tunParr1=histc(tmp,darr0);
tunParr1(end) = tunParr1(1);

tmp=[out(2).res(:).tuningP];
tmp=tmp(id2);
a2=tmp*2;
tunParr2=hist(tmp,darr0);
tunParr2(end) = tunParr2(1);

tmp=[out(3).res(:).tuningP];
tmp=tmp(id3);
a3=tmp*2;
tunParr3=hist(tmp,darr0);
tunParr3(end) = tunParr3(1);
%
ctr=0;
for nn=5:5:30
  bin=2*pi/nn;
  N=nn;

  ctr=ctr+1;
  ain=a1(~isnan(a1));
  M=length(ain);
  m=sum(abs(ain)<bin/2);
  pb1T(ctr)=binocdf(m,M,1/N)
  ain=a2(~isnan(a2));
  M=length(ain);
  m=sum(abs(ain)<bin/2);
  pb2T(ctr)=binocdf(m,M,1/N)
  ain=a3(~isnan(a3));
  M=length(ain);
  m=sum(abs(ain)<bin/2);
  pb3T(ctr)=binocdf(m,M,1/N)
end



pv1=circ_vtest(a1(~isnan(a1)),angle(sum(exp(i*a1(~isnan(a1))))))
pv2=circ_vtest(a2(~isnan(a2)),angle(sum(exp(i*a2(~isnan(a2))))))
pv3=circ_vtest(a3(~isnan(a3)),angle(sum(exp(i*a3(~isnan(a3))))))

subplot(3,2,4)
hold on
bar(darr0-.2,(tunParr1)/sum(tunParr1),'k','Barwidth',.3,'Edgecolor','w')
bar(darr0,(tunParr2)/sum(tunParr2),'g','Barwidth',.3,'Edgecolor','w')
bar(darr0+.2,(tunParr3)/sum(tunParr3),'r','Barwidth',.3,'Edgecolor','w')
hold off
set(gca,'Xlim',pi*[-1.2 1.2])
set(gca,'XTick',[-3:3:3])
set(gca,'Ylim',[0 .25]);
xlabel('Rel. heading (rad)')
ylabel('Norm. pair count')

text(angle(sum(exp(i*a1(~isnan(a1)))))/2,.215,num2str(pstr(pv1,2)), ...
     'Color','k')
text(angle(sum(exp(i*a2(~isnan(a2)))))/2,.175,num2str(pstr(pv2,2)), ...
     'Color','g')
text(angle(sum(exp(i*a3(~isnan(a3)))))/2,.175,num2str(pstr(pv3,2)), ...
     'Color','r')
%% Figure 1c
subplot(3,2,5)

narr=5:5:30;
hold on
plot(narr,pb1, '-k')
plot(narr,pb2, '-g')
plot(narr,pb3, '-r')

plot(narr,.05*ones(size(narr)), '-b')
hold off
set(gca,'Yscale', 'log')
set(gca,'Ylim', 10.^[-6 0])
xlabel('# heading bins')
ylabel('P value (dip test)')
%% Figure 1f (not)
subplot(3,2,6)
hold on
plot(narr,pb1T, '-k')
plot(narr,pb2T, '-g')
plot(narr,pb3T, '-r')
plot(narr,.05*ones(size(narr)), '-b')
%plot(0:.2:7,cumsum(tunKarr1)/sum(tunKarr1),'-k')
%plot(0:.2:7,cumsum(occKarr1)/sum(occKarr1),'--k')

%plot(0:.2:7,cumsum(tunKarr2)/sum(tunKarr2),'-g')
%plot(0:.2:7,cumsum(occKarr2)/sum(occKarr2),'--g')

%plot(0:.2:7,cumsum(tunKarr3)/sum(tunKarr3),'-r')
%plot(0:.2:7,cumsum(occKarr3)/sum(occKarr3),'--r')
hold off
%xlabel('Concentration k')
%ylabel('Cdf')
set(gca,'Yscale', 'log')
set(gca,'Ylim', 10.^[-6 0])
xlabel('# heading bins')
ylabel('P value (dip test)')

print('-depsc2','figpairs.eps')
%% EPS plot
!gv figpairs.eps&

dx=0.05
dy=0.05
posa1=[0.1 0.6 0.2 0.3]
posa2=posa1+[posa1(3)+dx 0 0 0];
posa3=posa2+[posa1(3)+dx 0 0 0];
posb1=posa1-[0 posa1(4)+dy 0 0];
posb2=posb1+[posa1(3)+dx 0 0 0];
posb3=posb2+[posa1(3)+dx 0 0 0];
%% FIgure 2 (radial plot)
figure(2);
clf

nbins=14
dphi=2*pi/nbins;
darr0=(0:nbins)*dphi-pi;

tmp=[out(1).res(:).tuningP];
rate1=histc(tmp,darr0);
rate1(end) = rate1(1);

tmp=[out(2).res(:).tuningP];
rate2=hist(tmp,darr0);
rate2(end) = rate2(1);

tmp=[out(3).res(:).tuningP];
rate3=hist(tmp,darr0);
rate3(end) = rate3(1);
%

aarr=[darr0];

rate1=rate1/sum(rate1)*length(aarr);
rate2=rate2/sum(rate2)*length(aarr);
rate3=rate3/sum(rate3)*length(aarr);

ratesym1=(rate1+rate1(end:-1:1))/2;
ratesym2=(rate2+rate2(end:-1:1))/2;
ratesym3=(rate3+rate3(end:-1:1))/2;

xcirc=cos(aarr)*pi;
ycirc=sin(aarr)*pi;

axes('Position',posa1)
hold on
plot(xcirc,ycirc,'-b');
plot((pi+amp1.*cos(aarr)).*cos(aarr),(pi+amp1.*cos(aarr)).*sin(aarr), '-k')
plot((pi+rate1).*cos(aarr),(pi+rate1).*sin(aarr), '--k')
plot([0 pi],[0 0], '-b')
plot([0 0],-[pi pi+length(aarr)/5], '-b')
hold off
axis square
set(gca,'visible','off')
set(gca,'XLim', [-1 1]*3*pi/2)
set(gca,'YLim', [-1 1]*3*pi/2)
text(pi/2,0.5,'\pi', 'Fontsize', 12, 'Color', 'b')
text(.5,-pi-1, '0.2', 'Fontsize', 12, 'Color', 'b')

axes('Position',posa2)
hold on
plot(xcirc,ycirc,'-b');
plot((pi+amp2.*cos(aarr)).*cos(aarr),(pi+amp2.*cos(aarr)).*sin(aarr), '-g')
plot((pi+rate2).*cos(aarr),(pi+rate2).*sin(aarr), '--g')
hold off
axis square
set(gca,'visible','off')
set(gca,'XLim', [-1 1]*3*pi/2)
set(gca,'YLim', [-1 1]*3*pi/2)

axes('Position',posa3)
hold on
plot(xcirc,ycirc,'-b');
plot((pi+amp3.*cos(aarr)).*cos(aarr),(pi+amp3.*cos(aarr)).*sin(aarr), '-r')
plot((pi+rate3).*cos(aarr),(pi+rate3).*sin(aarr), '--r')
hold off
axis square
set(gca,'visible','off')
set(gca,'XLim', [-1 1]*3*pi/2)
set(gca,'YLim', [-1 1]*3*pi/2)

axes('Position',posb1)
hold on
plot(xcirc,ycirc,'-b');
plot((pi+amp1.*cos(aarr)).*cos(aarr),(pi+amp1.*cos(aarr)).*sin(aarr), '-k')
plot((pi+ratesym1).*cos(aarr),(pi+ratesym1).*sin(aarr), '--k')
hold off
axis square
set(gca,'visible','off')
set(gca,'XLim', [-1 1]*3*pi/2)
set(gca,'YLim', [-1 1]*3*pi/2)

axes('Position',posb2)
hold on
plot(xcirc,ycirc,'-b');
plot((pi+amp2.*cos(aarr)).*cos(aarr),(pi+amp2.*cos(aarr)).*sin(aarr), '-g')
plot((pi+ratesym2).*cos(aarr),(pi+ratesym2).*sin(aarr), '--g')
hold off
axis square
set(gca,'visible','off')
set(gca,'XLim', [-1 1]*3*pi/2)
set(gca,'YLim', [-1 1]*3*pi/2)

axes('Position',posb3)
hold on
plot(xcirc,ycirc,'-b');
plot((pi+amp3.*cos(aarr)).*cos(aarr),(pi+amp3.*cos(aarr)).*sin(aarr), '-r')
plot((pi+ratesym3).*cos(aarr),(pi+ratesym3).*sin(aarr), '--r')
hold off
axis square
set(gca,'visible','off')
set(gca,'XLim', [-1 1]*3*pi/2)
set(gca,'YLim', [-1 1]*3*pi/2)

print('-depsc2','figpairssc.eps')
!gv figpairssc.eps
