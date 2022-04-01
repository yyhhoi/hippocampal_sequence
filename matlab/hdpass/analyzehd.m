clear
load('hd.mat')

nca1=0;nca2=0;nca3=0;
CA1.R=[];CA1.pval=[];CA1.N=[];CA1.a=[];CA1.r=[];
CA2.R=[];CA2.pval=[];CA2.N=[];CA2.a=[];CA2.r=[];
CA3.R=[];CA3.pval=[];CA3.N=[];CA3.a=[];CA3.r=[];
passes(6).CA1=[];passes(6).CA2=[];passes(6).CA3=[];
for nd=1:length(day)
  trials=day(nd).trials;
  
  for nt=1:length(trials)
    
    if isfield(trials(nt),'CA1fields')
      na=length(trials(nt).CA1fields);
      if na>0
        nca1=nca1+na;
        
        CA1.R=[CA1.R [trials(nt).CA1fields(:).RVL]];
        CA1.r=[CA1.r [trials(nt).CA1fields(:).peakrate]];
        CA1.a=[CA1.a [trials(nt).CA1fields(:).area]];
        CA1.pval=[CA1.pval [trials(nt).CA1fields(:).pval]];
        CA1.N=[CA1.N [trials(nt).CA1fields(:).N]];
        
        for nf=1:na
          for nth=1:6
            passes(nth).CA1=[passes(nth).CA1 pcheckout(trials(nt).CA1fields(nf).passes(nth))];
          end
        end
      end
    end
    if isfield(trials(nt),'CA2fields')
      na=length(trials(nt).CA2fields);
      if na>0
        nca2=nca2+na;
        
        CA2.R=[CA2.R [trials(nt).CA2fields(:).RVL]];
        CA2.pval=[CA2.pval [trials(nt).CA2fields(:).pval]];
        CA2.r=[CA2.r [trials(nt).CA2fields(:).peakrate]];
        CA2.a=[CA2.a [trials(nt).CA2fields(:).area]];
        CA2.N=[CA2.N [trials(nt).CA2fields(:).N]];
        for nf=1:na
          for nth=1:6
            passes(nth).CA2=[passes(nth).CA2 ...
                        pcheckout(trials(nt).CA2fields(nf).passes(nth))];
          end
        end
      end
    end

    if isfield(trials(nt),'CA3fields')
      na=length(trials(nt).CA3fields);
      if na>0
        nca3=nca3+na;
        
        CA3.R=[CA3.R [trials(nt).CA3fields(:).RVL]];
        CA3.pval=[CA3.pval [trials(nt).CA3fields(:).pval]];
        CA3.N=[CA3.N [trials(nt).CA3fields(:).N]];
        CA3.r=[CA3.r [trials(nt).CA3fields(:).peakrate]];
        CA3.a=[CA3.a [trials(nt).CA3fields(:).area]];
        for nf=1:na 
          for nth=1:6
            passes(nth).CA3=[passes(nth).CA3 pcheckout(trials(nt).CA3fields(nf).passes(nth))];
          end
        end

      end
    end
    
  end

end

[nca1 nca2 nca3]

  
tarr=[0:100:1200];
for ncntr=1:length(tarr)-1
  nthresh=tarr(ncntr);
  nthreshu=tarr(ncntr+1);
  id1= find(CA1.N>=nthresh & ~isnan(CA1.R) & CA1.a>300);% & CA1.N<nthreshu);
  id2= find(CA2.N>=nthresh & ~isnan(CA2.R) & CA2.a>300);% & CA2.N<nthreshu);
  id3= find(CA3.N>=nthresh & ~isnan(CA3.R) & CA3.a>300);% & CA3.N<nthreshu);


  fsig(ncntr,1)=sum(CA1.pval(id1)<0.05)/length(id1);
  fsig(ncntr,2)=sum(CA2.pval(id2)<0.05)/length(id2);
  fsig(ncntr,3)=sum(CA3.pval(id3)<0.05)/length(id3);
  
  md(ncntr,1)=nanmedian(CA1.R(id1));
  md(ncntr,2)=nanmedian(CA2.R(id2));
  md(ncntr,3)=nanmedian(CA3.R(id3));

  prop(ncntr,1)=length(id1)/sum(~isnan(CA1.R) & CA1.a>300);
  prop(ncntr,3)=length(id3)/sum(~isnan(CA3.R) & CA3.a>300);
  prop(ncntr,2)=length(id2)/sum(~isnan(CA2.R) & CA2.a>300);
  
  
  for nth=1:6
    
    Rt1=[passes(nth).CA1.RVL];
    Rt2=[passes(nth).CA2.RVL];
    Rt3=[passes(nth).CA3.RVL];
    
    id1= find([passes(nth).CA1.N]>=nthresh & ~isnan(Rt1));
    id2= find([passes(nth).CA2.N]>=nthresh & ~isnan(Rt2));
    id3= find([passes(nth).CA3.N]>=nthresh & ~isnan(Rt3));
    
    pt1=[passes(nth).CA1.pval];
    pt2=[passes(nth).CA2.pval];
    pt3=[passes(nth).CA3.pval];
    passes(nth).fsig(ncntr,1)=sum(pt1(id1)<0.05)/length(id1);
    passes(nth).fsig(ncntr,2)=sum(pt2(id2)<0.05)/length(id2);
    passes(nth).fsig(ncntr,3)=sum(pt3(id3)<0.05)/length(id3);
   
    passes(nth).md(ncntr,1)=nanmedian( Rt1(id1));
    passes(nth).md(ncntr,2)=nanmedian( Rt2(id2));
    passes(nth).md(ncntr,3)=nanmedian( Rt3(id3));
    
    passes(nth).prop(ncntr,1)=length(id1)/sum(~isnan( Rt1));
    passes(nth).prop(ncntr,2)=length(id2)/sum(~isnan( Rt2));
    passes(nth).prop(ncntr,3)=length(id3)/sum(~isnan( Rt3));
    
  end
end

thresharr=[-10 0 1 3 5 7];

figure(2)
clf

subplot(2,2,1)
hold on
for nth=[1 5]%1:length(thresharr)
  
  plot(tarr(2:end), passes(nth).md(:,1),'-k', 'Color',[1 1 1]*(nth-1)/5*.4)
  plot(tarr(2:end), passes(nth).md(:,2),'-g', 'Color',[0 (nth-1)/5*.5+.5 ...
       0])
  plot(tarr(2:end), passes(nth).md(:,3),'-r', 'Color',[(nth-1)/5*.5+.5 0 ...
       0])

end
hold off
xlabel('Nthresh')
ylabel('Median R')

subplot(2,2,2)
hold on
for nth=[1]%1:length(thresharr)
  
  plot(tarr(2:end), passes(nth).fsig(:,1),'-k', 'Color',[1 1 1]*(nth-1)/5*.4)
  plot(tarr(2:end), passes(nth).fsig(:,2),'-g', 'Color',[0 (nth-1)/5*.5+.5 ...
       0])
  plot(tarr(2:end), passes(nth).fsig(:,3),'-r', 'Color',[(nth-1)/5*.5+.5 0 ...
       0])

end
hold off
xlabel('Nthresh')
ylabel('Fract. significant')

subplot(2,2,3)
hold on
for nth=[1 5]%1:length(thresharr)
  
  plot(tarr(2:end), passes(nth).prop(:,1),'-k', 'Color',[1 1 1]*(nth-1)/5*.4)
  plot(tarr(2:end), passes(nth).prop(:,2),'-g', 'Color',[0 (nth-1)/5*.5+.5 ...
       0])
  plot(tarr(2:end), passes(nth).prop(:,3),'-r', 'Color',[(nth-1)/5*.5+.5 0 ...
       0])

end
hold off
xlabel('Nthresh')
ylabel('Data fraction')

subplot(2,2,4)
hold on
for nth=[5]%1:length(thresharr)
  
 plot(tarr(2:end), passes(nth).fsig(:,1),'-k', 'Color',[1 1 1]*(nth-1)/5*.4)
  plot(tarr(2:end), passes(nth).fsig(:,2),'-g', 'Color',[0 (nth-1)/5*.5+.5 ...
       0])
  plot(tarr(2:end), passes(nth).fsig(:,3),'-r', 'Color',[(nth-1)/5*.5+.5 0 ...
       0])


end
hold off
xlabel('Nthresh')
ylabel('Fract. significant')

figure(1)
clf

subplot(2,2,1)
hold on
plot(tarr(2:end),md(:,1),'-k')
plot(tarr(2:end),md(:,2),'-g')
plot(tarr(2:end),md(:,3),'-r')
hold off
xlabel('Nthresh')
ylabel('Median R')

subplot(2,2,2)
hold on
plot(tarr(2:end),fsig(:,1),'-k')
plot(tarr(2:end),fsig(:,2),'-g')
plot(tarr(2:end),fsig(:,3),'-r')
hold off
xlabel('Nthresh')
ylabel('Fract. significant')

subplot(2,2,3)
hold on
plot(tarr(2:end),prop(:,1),'-k')
plot(tarr(2:end),prop(:,2),'-g')
plot(tarr(2:end),prop(:,3),'-r')
hold off
xlabel('Nthresh')
ylabel('Data fraction')

subplot(2,2,4)
hold on
plot(CA1.r,CA1.R,'.k')
plot(CA2.r,CA2.R,'.g')
plot(CA3.r,CA3.R,'.r')
hold off
xlabel('Rate')
ylabel('RVL')
