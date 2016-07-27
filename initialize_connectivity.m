function [cardiac_nodes,neigh_nodes,connectivity,...
          nodes_state,nodes_voltage,nodes_coor]=initialize_connectivity(W)
      
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Simulaci??n del tejido cardiaco en 2D, por medio del modelo probabil??stico  
% basado en un aut??mata celular 
%  Fecha programa:   2012, 
%  Autores:         Ferney Beltr??n-Molina    Jes??s Requea Carri??n
%                   ferney.beltran@gmail.com jesus.requena@urjc.es
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% # nodes tissue: WxW  
number_nodes =  W*W;    
nodes_coor   =  zeros(number_nodes,3);
indices_heart   = (int32(1:number_nodes));
                            
%% # nodes neighbours
N_neigh=4;  % 4 for 2D [x+1; y+1; x-1; y-1]
            % 6 for 3D [x+1; y+1; z+1; x-1; y-1; z-1]

per_neigh =     number_nodes+1;
indices_neigh_rel   = int32(zeros(number_nodes,N_neigh));

%para hacer que el shifting no sea circular  ／／ 为了失移不会圆移
indices_sharp = int32(ones(W+2,W+2)*per_neigh); 
indices_sharp(2:end-1,2:end-1) = reshape(indices_heart,W,W);

shiftsize = [-1 0; 0 -1; 1 0; 0 1 ...
            ;-1 -1;1 -1; -1 1;1 1];
for i=1:N_neigh
    neigh_t = circshift(indices_sharp, shiftsize(i,:));
    indices_neigh_rel(:,i) = reshape(neigh_t(2:end-1,2:end-1),1,[]);
end

%
% indices_neigh_rel(:,[1 3])=10202;
%

cardiac_nodes = indices_heart';
neigh_nodes = indices_neigh_rel;
nodes_all   = [cardiac_nodes ;per_neigh];

%% # connectivity
% Setting to 0 connectivities to nodes not belonging to the ventricles and
% conductivities from nodes in the atria.

cond_no=0.0050;
cond_pe=cond_no/2;

connectivity_prov = ones(number_nodes,N_neigh)*cond_no;

for i=1:N_neigh
   nodes_no_heart=(neigh_nodes(:,i)~=per_neigh);   
   connectivity(:,i)=connectivity_prov(:,i).*nodes_no_heart;
end

%%
nodes_state=zeros(size(nodes_all));  %<10202 by 1>
nodes_voltage=-60*ones(size(nodes_all));

%% coordinates
lX= 1:W;
Temp_coor = repmat(lX,W,1);
nodes_coor(:,1)=Temp_coor(:);
Temp_coor=repmat(lX',1,W);
nodes_coor(:,2)=Temp_coor(:);

