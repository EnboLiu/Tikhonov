function  [EGM, VT]  = simulate_CAM(class,W,W_e,sim_time,varargin)


%
%
%
%   Simulation of cardiac dynamics based on a cellular automata model (CAM)
%
%
%   Ref: F Alonso-Atienza, J Requena-Carrion et al., 'A Probabilistic Model 
%       of Cardiac Electrical Activity Based on a Cellular Automata System'
%
%
%   Created:    2003 
%   Authors:    Jesus Requena Carrion, j.requena@qmul.ac.uk
%
%   Modified:   2013  
%   Authors:    Ferney Beltran-Molina, ferney.beltran@gmail.com
%                    
%   Modified:   2014 
%   Authors:    Ferney Beltran-Molina, ferney.beltran@gmail.com
%               - New restitution curves
%
%
%   
%   'class' describes the type of stimulation
%           1) 'plane': plane wavefronts, for regular rhythms
%           2) 'S1-S2': cross-stimulation, for spiral dynamics
%
%   'W' defines the dimensions of the the 2D square tissue sample: W x W       
%           By default W=101
%
%   'W_e' defines the dimensions of the array of electrodes: W_e x W_e
%           By default W_e=4
%
%   'sim_time' is the simulated time
%           By default sim_time=2 s
%
%   For the remaining input arguments (not described) see below:
%
%       simulate_EGM(class,W,W_e,sim_time, ...
%                   'PropertyName',PropertyValue ...)
%
%       The 'PropertyName'/PropertyValue pairs can be any of the following
%                    'StimulationTimes', pulse_array ... 
%                    'SamplingFreq', FreqValue ...
%                    'ConstRestitution', [cAPD [cCV]]  ...
%                    'StimulationZones', stim_nodes ...
%                    'StimulationSequence', 'random' or 'cyclic' ...
%                    'DeadZone', inactive_region ... 
%                    'MSD_model', transfer_matrix ...                       
%



clc
act_path  = path;
addpath(genpath(pwd));


%% INITIALIZATION

%   ---------------------
%   SIMULATION PARAMETERS
%   ---------------------

if mod(length(varargin),2), error('Initialization error');end
if nargin == 0, class = 'plane'; end
if nargin < 2, W = 34; end 
if nargin < 3, W_e = 10; end 
if nargin < 4, sim_time  = 2; end

% Tissue dimensions
tissue_size = W*W;

% Time step in simulations 
FsSim   = 1000;
Dt      = 1/FsSim;           
t       = 0;

% Sampling frequency for saving signals and videos
FsSignal    = SetParameter(1000,'SamplingFreq',varargin{:});
FsFrame     = FsSignal;
Frame_index = 1;


%   --------------
%   CARDIAC TISSUE
%   --------------


%   Geometry and basic connectivity
[cardiac_nodes,neigh_nodes,connectivity,nodes_state,nodes_voltage, nodes_coor]  = ...
    initialize_connectivity(W);

%   Inactive regions
inactive_region     = reshape(SetParameter(0,'DeadZone',varargin{:}),1,[]);
for inds_inactive   = inactive_region
    connectivity    = connectivity.*(neigh_nodes ~= inds_inactive);
end

%   Transmembrane potentials
TMP                 = zeros(tissue_size,sim_time*FsFrame+1);

%   Reference action potential
load Vref;
length_vref     = length(Vref);        

%   Cardiac restitution
cAPD    = SetParameter(1.0,'ConstRestitution',varargin{:});
if length(cAPD)==1, cCV=cAPD; else cCV = cAPD(2); cAPD=cAPD(1);end
K       = 0.9; 
F_APD   = 0.1;       
t_desp  = -0.4*ones(tissue_size,1);           
t_rep   = -0.2*ones(tissue_size,1);
rAPD    = 0.2*ones(size(t_rep));

%   Stimuation times
pulse_array     = SetParameter(((0:0.5:sim_time)),'StimulationTimes',varargin{:});
if strcmp(class, 'S1-S2'),pulse_array = [0 0.135]; end
stim_times      = [sort(pulse_array) nan];
stim_index      = 1;

%   Stimulus configuration
if strcmp(class, 'plane')
    stim_nodes = stimulation_region(W,class,1,1);
    stim_nodes_index = ones(length(pulse_array),1);
elseif strcmp(class, 'S1-S2')
    stim_nodes = stimulation_region(W,class,round(W/2),2);
    stim_nodes_index = [1 2];
else
    stim_nodes = SetParameter(0,'StimulationZones',varargin{:});
    [sm, N] = size (stim_nodes); 
    if (sm ~= tissue_size)
        error('The dimensions of StimulationZones must be (W*W,N)');
    end
    ss = SetParameter('random','StimulationSequence',varargin{:});
    if strcmp(ss, 'cyclic')
        stim_nodes_index = 1:N;
        factor =ceil (length(pulse_array)/length(stim_nodes_index));
        if factor>1
            stim_nodes_index = repmat(stim_nodes_index,1,factor);
            stim_nodes_index = stim_nodes_index(1:length(pulse_array));
        end
    else  
        stim_nodes_index = randi(N,size(pulse_array));
        
        if N>2
            stim_nodes_index(2:end+1)= stim_nodes_index; 
            stim_nodes_index(1)=0; 
            ind_seg= find((stim_nodes_index(2:end)-stim_nodes_index(1:end-1))==0);

            for i=1:length(ind_seg)
                pos = ind_seg(i);
                NV=stim_nodes_index(pos);
                while (NV==stim_nodes_index(pos)) || (NV==stim_nodes_index(pos-1))
                    NV=random('unid',N);
                end
                stim_nodes_index(pos)=NV;
            end
            stim_nodes_index=stim_nodes_index(2:end); 
        end
    end
    
end

if (length(pulse_array) > length(stim_nodes_index)) 
    error('StimulationTime: Wrong dimensions');end



%   ------------------
%   MEASUREMENT SYSTEM
%   ------------------

%   Array of unipolar electrodes

h_e                 = 5;
le                  = round(W/(2*W_e)):round(W/W_e):W;
Temp_coor           = repmat(le,W_e,1);
Electrode_coor(:,1) = Temp_coor(:);
Temp_coor           = repmat(le',1,W_e);
Electrode_coor(:,2) = Temp_coor(:);
Electrode_coor(:,3) = h_e*ones(size(Electrode_coor(:,2)));
N_electrode     = size(Electrode_coor,1);
transfer_matrix = zeros(N_electrode,tissue_size);

for l=1:N_electrode
    
    XE1=nodes_coor(:,1)-Electrode_coor(l,1);
    YE1=nodes_coor(:,2)-Electrode_coor(l,2);
    ZE1=nodes_coor(:,3)-Electrode_coor(l,3);
    
    H1_E1=sqrt(XE1.^2+YE1.^2);
    H_E1=sqrt(H1_E1.^2 +ZE1.^2);

    Theta=acos(H1_E1./H_E1); 
    Phi=acos(XE1./H1_E1).*(sign(YE1));
    Phi(isnan(Phi))=pi;
    
    modLF= 1./(H_E1.^2);
    modxyLF= cos(Theta).*modLF;
    zLF= sin(Theta).*modLF;

    modxyLF=modxyLF/max(max(modxyLF));

    arrayMLFx= cos(Phi).*modxyLF;
    arrayMLFy= sin(Phi).*modxyLF;
%    arrayMLFz= zLF;
    
    arrayMLFx_ext=[arrayMLFx; 0];
    arrayMLFy_ext=[arrayMLFy; 0];
    arrayILF = arrayMLFx-arrayMLFx_ext(neigh_nodes(:,1))+arrayMLFy-arrayMLFy_ext(neigh_nodes(:,2));
    transfer_matrix(l,:) = arrayILF;     
end

CardiacSignal   = zeros(size(transfer_matrix,1),sim_time*FsSignal+1);



%   ------------------
%   INVERSE SOLUTION
%   ------------------


%A   = transfer_matrix;


%{
ncA = size(A,2);
L   = eye(ncA,ncA); 
L2  = L*L';
A2  = A'*A;
lambda  = 0.002;
%}



%M_inv=inv(A2+topt*L2)*A';

%{
B       = A2+lambda*L2;
M_inv   = B\A.';
%}


VIP     = zeros(size(TMP));


%%   SIMULATION

tic

while t<sim_time
    
    % State transitions
    Ex_nodes            = (nodes_state(cardiac_nodes)==2);
    Ref_nodes           = (nodes_state(cardiac_nodes)==1);
    Rest_nodes          = (nodes_state(cardiac_nodes)==0);
     
    nodes_excitability  = cv_restitution((t-t_rep)*1000,cCV).*Rest_nodes;
    nodes_excitation    = sum((nodes_state(neigh_nodes)==2).*connectivity,2);
    p_exc               = K*nodes_excitation.*nodes_excitability;

    Rest_transition     = (rand(size(p_exc))<p_exc).*Rest_nodes;        
    Ex_transition       = Ex_nodes.*((t-t_desp)>(F_APD*rAPD));    
    Ref_transition      = Ref_nodes.*((t-t_desp)>rAPD);

    nodes_state(cardiac_nodes) = nodes_state(cardiac_nodes)+...
        2*Rest_transition-Ref_transition-Ex_transition;       
    
    % External stimulus
    if (t>stim_times(stim_index))
    
        Rest_transition  = stim_nodes(:,stim_nodes_index(stim_index));
        stim_index = stim_index + 1;
    
        nodes_state(cardiac_nodes) = nodes_state(cardiac_nodes).* ...
           not(Rest_transition)+2*Rest_transition;
    end
    
    t_rep  = t*Ref_transition+t_rep.*not(Ref_transition);
    t_desp = t*Rest_transition+t_desp.*not(Rest_transition);        
    
    rAPD = apd_restitution((t_desp-t_rep)*1000,cAPD).*Rest_transition+...
         rAPD.*not(Rest_transition);
    
     
    % Measurements 
    t_mes=round(t/Dt);
    
    if (mod(t_mes,FsSim/FsSignal)==0)
        clc, disp([num2str(t) ' of ' num2str(sim_time) ' sec.']);
        indicesV = ceil(1+(length_vref-1)*trapmf(...
            (t-t_desp)*0.180*10000./rAPD+1,[0 length_vref inf inf]));  
        nodes_voltage(cardiac_nodes)=Vref(indicesV);
        V = nodes_voltage(cardiac_nodes);%************************
        % this is the to be got X`
        V_frame(:,Frame_index)= V;
        
        
        
        %{ Tikhonov regularization with 0-order
        CardiacSignal(:,Frame_index) = transfer_matrix*V;%**********
        % this is the signal acquired as the  GOT b
       
        b = transfer_matrix*V;        
        
        A   = transfer_matrix;
        
        lambda = 3.00;
        
        tissuesize = size(A,2);
        
        A = [A ; lambda^2*eye(tissuesize)];
        
        b_b = [b ; zeros(tissuesize,1)];
        
        Xe = lsqr(A,b_b);
        
        
        if (mod(t_mes,FsSim/FsFrame)==0)
            if t_mes>0, TMP(:,round(t_mes/(FsSim/FsFrame))) = V; end
        end
        
        %----------
        
        %VIPtemp=M_inv*CardiacSignal(:,Frame_index);
        
        VIPtemp = Xe - 80;
        
        VIP(:,Frame_index)  = VIPtemp;
        
       errors(1,Frame_index) = the_error(V,Xe);    %addition
        
        Frame_index         = Frame_index+1;
        
    end
    
    t   = t+Dt;
end

%plot(errors);
%axis([0 2000 0.99 1]);

toc

save('errors.mat','errors');

%%   VIDEO

if nargout==0
    if ~isempty(transfer_matrix)
        video_CAM(TMP,VIP,transfer_matrix(:,1),CardiacSignal(1,:),FsSignal,FsSignal/10);
    else
        video_CAM(TMP,VIP,[],CardiacSignal(1,:),FsSignal,FsSignal/10);
    end
else
    EGM = CardiacSignal;   
end
if nargout>=2, VT=TMP; end
path(act_path);




function pvo = SetParameter (DefaultValue,ParameterName,varargin)  
  
    Pflag=find(strcmp(ParameterName,varargin));
    if any(Pflag)
        % Extract flag
        pv = varargin{Pflag+1};
        % si el tipo (numeric or char) del valor pasado es igual al esperado 
        if xor(isnumeric(pv),isnumeric(DefaultValue)) 
            pvo =  DefaultValue; 
            error([ 'error al definir los valores de ' ParameterName]);
        end
            
    else
        pv  =  DefaultValue;    
    end
    pvo     = pv;
    
    
function vAPD = apd_restitution(DI,cAPD)
    Tau_apd = 25;
    vAPD    = cAPD*0.0700*(1-exp(-DI/Tau_apd));

 
    
function vCV = cv_restitution(DI,cVC)
    Tau_cv  = 35;
    vCV     = cVC*60*(1-exp(-DI/Tau_cv));


function error_character = the_error(v_real,v_estimated)
    error_character = sum((v_real - v_estimated).^2)/sum(v_real.^2);
   