function nod_ect_array = stimulation_region(W,class,lside,N,offset)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Authors:         Ferney Beltran-Molina    Jesus Requea Carrion
%                   ferney.beltran@gmail.com j.requena@qmul.ac.uk
%                   june 2012


nod_ect_array  = zeros(W*W,N,'single'); %w=3,n=2,72elementes

if strcmp(class, 'plane')
    nod_ect_array(1:W*lside,:)=1;
else
    temp_nod_ect = zeros(W);  
    temp_nod_ect(1:lside,1:lside)  = 1; 
    switch class
        case 'S1-S2'
            nod_ect_array(1:W,1)  = 1;
            nod_ect_array(:,2) = reshape(temp_nod_ect,1,[]);
        case 'distri'
            if ~exist('offset','var')
                offset=0;
            end
            nod_ect_array = distri(W,lside,N,offset);
        case 'random'
            for ni=1:N  
                cx = randi(W-lside);
                cy = randi(W-lside);
                nod_ect_array(:,ni) = ... 
                    reshape(circshift(temp_nod_ect, [cx cy]),1,[]);
            end
        otherwise
            
    end
end
if ~nargout
    figure; hold on
    for nT=1:size(nod_ect_array,2)
%         figure
        V = reshape(double(nod_ect_array(:,nT)),W,W);
        surf(0:W-1,0:W-1,V)
        view(0,90)
        shading interp
        axis square
        caxis([0 1]);
        colorbar
        axis([0 W-1 0 W-1]);
    end
end 

function nod_ect_array = distri(W,lside,N,offset)
    
    nod_ect_array  = zeros(W*W,N*N,'single');
    
    if N>=(W-lside)
        N=(W-lside-1);
    end
    ni=0;

    lsidex=lside(1);
    if length(lside)==2, lsidey=lside(2); else lsidey =lsidex; end
    temp_nod_ect = zeros(W);
    temp_nod_ect(1:lside,1:lsidey)  = 1;
    [lshiftx mincx] =  separa(W,lsidex,N);
    [lshifty mincy] =  separa(W,lsidey,N);
    
    for cx=mincx:lshiftx:W-lsidex
        for cy=mincy:lshifty:W-lsidey
            ni=ni+1;
            nod_ect_array(:,ni) = ... 
            reshape(circshift(temp_nod_ect, [cx cy-offset]),1,[]);
        end
    end
    
function [lshift minc] =  separa(W,lside,N)
    minc = round((W-N*lside)/N);
    lshift =lside+minc;
    minc =round(minc/2);
    if minc<0, minc=0; lshift=lshift-1;end 