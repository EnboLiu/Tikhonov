function video_CAM(V,VIP,MSD,EGM, FsE, FsV)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Autores:         Ferney Beltran-Molina    Jesus Requea Carrion
%                   ferney.beltran@gmail.com jesus.requena@urjc.es
%                   junio 2012
%
% Modified Nov 2014 (JRC): avifile removed, VideoWriter used
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



mov=VideoWriter('Animation.avi');
open(mov);

W= sqrt(size(V,1));

x = 0:W-1;
y = x;



EGM= EGM./max(abs(EGM));
EGM_max = max(EGM);
EGM_min = min(EGM);

fa = figure;

allTime    = length(EGM)/FsE;
array_time = (0:1/FsE:allTime-1/FsE);

% Plot cardiac signal
subplot(313)
plot(array_time,EGM);
grid on;


mf=round(length(EGM)/length(V(1,:)))-1;
if mf<1, mf=1;end

if exist ('FsV','var')
    nn= abs(round(FsE/FsV));
else
    nn=1;
end
mm = min(min(V));
mx = max(max(V));
mmIP = min(min(VIP));
mxIP = max(max(VIP));


for nT=1:nn:length(V(1,:))-10
    
    n_t =mf*nT+1;
    if n_t > length(array_time)
        n_t = length(array_time);
        nT = round((n_t-1)/mf);
    end
    array_time(n_t);
    
    figure(fa);subplot(313)
    if n_t< (length(array_time)-round(5/array_time(2)))
         axis([array_time(n_t) array_time(n_t+round(5/array_time(2))) EGM_min  EGM_max]);
    end
    title([num2str(round(1000*array_time(n_t))),' ms'  ])

    figure(fa);subplot(221)
    Vsurf = reshape(double(V(:,nT)),W,W);
    surf(x,y,Vsurf)
    view(0,90)    %view(45,45)
    shading interp
    axis square
    caxis([mm mx]);
    colorbar
    axis([0 W-1 0 W-1]);
    title('Simulated voltage')
    
    figure(fa);subplot(222)
    Vsurf = reshape(double(VIP(:,nT)),W,W);% seems it should be VIP
    surf(x,y,Vsurf)
    view(0,90)   
    shading interp
    axis square
    caxis([mmIP mxIP]); %caxis([-80 40])  ;% some should be modified  %caxis([mmIP mxIP])
    colorbar
    axis([0 W-1 0 W-1]);
    title('Estimated (IP) voltage')
    
    frame=getframe(fa);
    writeVideo(mov,frame);
    
    
 end    
 close(mov);
%-----------------------------
