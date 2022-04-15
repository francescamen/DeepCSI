%% beamforming_matrix_analysis_quantization.m

% Copyright (C) 2022 Francesca Meneghello
% contact: meneghello@dei.unipd.it
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
% You should have received a copy of the GNU General Public License
% along with this program.  If not, see <https://www.gnu.org/licenses/>.

clear all
addpath('IndoorCommLinkUsingRayTracingExample/');

%% configuration
Nc = 2;  % number of spatial streams
Nr = 3;  % number of Tx antennas

%% MIMO OFDM CHANNEL MODEL (from IndoorCommLinkUsingRayTracingExample.mlx)
fc = 5210e6;
lambda = physconst('lightspeed')/fc;

%% 3D Indoor Scenario
mapFileName = "conferenceroom.stl";
txArray = arrayConfig("Size", [3 1], 'ElementSpacing', lambda/2);
rxArray = arrayConfig("Size", [2 1], 'ElementSpacing', 2*lambda);
tx = txsite("cartesian", ...
    "Antenna",txArray, ...
    "AntennaPosition",[-1.46; -1.42; 2.1], ...
    'TransmitterFrequency',fc);
rx = rxsite("cartesian", ...
    "Antenna",rxArray, ...
    "AntennaPosition",[.3; .3; .85], ...
    "AntennaAngle",[0;90]);
helperVisualizeScenario(mapFileName,tx,rx);

%% Ray Tracing
max_order_reflections = 4;
pm = propagationModel("raytracing", ...
    "CoordinateSystem","cartesian", ...
    "Method","sbr", ...
    "AngularSeparation","low", ...
    "MaxNumReflections",max_order_reflections, ...
    "SurfaceMaterial","wood");
rays = raytrace(tx,rx,pm,'Map',mapFileName);
rays = rays{1,1};
num_reflections = [rays.NumReflections];
propagation_distance = [rays.PropagationDistance];
path_loss = [rays.PathLoss];
propagation_delay = [rays.PropagationDelay];
aoa = [rays.AngleOfArrival];
helperVisualizeRays(rays);

%% Deterministic Channel Model from Ray Tracing
rtChan = comm.RayTracingChannel(rays, tx, rx);
rtChan.SampleRate = 80e6; 
rtChan.ReceiverVirtualVelocity = [0.01; 0; 0];
showProfile(rtChan);

rtChanInfo = info(rtChan);
disp(rtChanInfo);
numTx = rtChanInfo.NumTransmitElements;
numRx = rtChanInfo.NumReceiveElements;

%% Communication link
% Create LDPC encoder and decoder objects
ldpcEnc = comm.LDPCEncoder;
ldpcDec = comm.LDPCDecoder;
numCodewordsPerFrame = 1;
codewordLen = size(ldpcEnc.ParityCheckMatrix, 2);
disp(['LDPC parity check matrix size: ', num2str(size(ldpcEnc.ParityCheckMatrix))]);

%% Parameteres 
fftLen = 256; 
cpLen = fftLen/4; 
numGuardBandCarriers = [6; 5];
pilotCarrierIdx = [-103, -75, -39, -11, -1, 1, 11, 39, 75, 103]';
pilotCarrierIdx = pilotCarrierIdx + 129;
numDataCarriers = fftLen - sum(numGuardBandCarriers) - length(pilotCarrierIdx) - 1;

%% Link Simulation
numFrames = 100000;  
chanEstArray = zeros(numDataCarriers, numFrames, numTx, numRx);
for fr = 1:numFrames

    [~, CIR] = rtChan(ones(1,3)); 

    %% CFR
    % Perfect channel estimation
    dataCarrierIdx = setdiff( ...
        numGuardBandCarriers(1)+1:fftLen-numGuardBandCarriers(2), ...
        [pilotCarrierIdx; fftLen/2+1]);
    chanDelay = channelDelay(CIR, rtChanInfo.ChannelFilterCoefficients);
    chanEst = helperPerfectChannelEstimate( ...
        CIR, rtChanInfo.ChannelFilterCoefficients, fftLen, cpLen, ...
        dataCarrierIdx, chanDelay, 'OFDMSymbolOffset', .8);  
    chanEstArray(:, fr, :, :) = chanEst;
end
CFR = permute(chanEstArray, [2, 3, 1, 4]);

stream_n = 2;
ant_n = 1;
figure();
h = pcolor(squeeze(abs(chanEstArray(:, :, ant_n, stream_n))).');
set(h, 'EdgeColor', 'none');
colorbar;

figure();
for i = 1:10000
    plot(squeeze(abs(chanEstArray(:, i, ant_n, stream_n))));
    hold on
end

%% CLEAR UNUSED VARIABLES
clear aoa chanEst chanEstArray CIR chanIn dataCarrierIdx errRate ldpcDec 
clear ldpcEnc num_reflections ofdmDemod ofdmMod path_loss pm 
clear propagation_delay propagation_distance rays rtChan rtChanInfo rx 
clear rxArray srBits tx txArray txWave

%% COMPUTE V AND BEAMFORMING FEEDBACK
psi_bit = 7;
phi_bit = psi_bit + 2;

phi_numbers = 3;
psi_numbers = 3;
tot_angles = phi_numbers + psi_numbers;
order_angles = ['phi_11', 'phi_21', 'psi_21', 'psi_31', 'phi_22', 'psi_32'];
order_bits = [phi_bit, phi_bit, psi_bit, psi_bit, phi_bit, psi_bit];

const1_phi = 2^(phi_bit-1);
const2_phi = 2^(phi_bit);

const1_psi = 2^(psi_bit+1);
const2_psi = 2^(psi_bit+2);

V_matrices = {};
Vreconstruct_matrices  = {};

n = size(CFR, 1);
k = 1;
while (k <= n)
    H_matrix = squeeze(CFR(k, :, :, :));
    V_matrix = zeros(Nr, numDataCarriers, Nc);
    Vreconstruct_matrix  = zeros(Nr, numDataCarriers, Nc);
    
    for s_i = 1:numDataCarriers
        H = squeeze(H_matrix(:, s_i, :));
        H = H.';
        
        [U,S,V] = svd(H);
        
        V = V(:, 1:Nc);
        V_matrix(:, s_i, :) = V;
        
        D_i_matrices = {};
        G_li_matrices = {};
        
        vm_angles_vector = angle(V(Nr, :)); 
        Dtilde = diag(exp(1i*vm_angles_vector));
        Omega = V * Dtilde';
        for i = 1:min(Nc, Nr-1)
            %% compute phi and build D matrix
            D_i = eye(Nr);
            for l = i:Nr-1
                phi_li = angle(Omega(l, i));
                
                quantized_phi_li = quantize(phi_li, const1_phi, const2_phi);
                phi_li_rad = inverse_quantize(quantized_phi_li, const1_phi, const2_phi);

                D_i(l, l) = exp(1i*phi_li_rad);
            end
            D_i_matrices{i} = D_i;
            Omega = D_i' * Omega;
            
            %% compute psi and build G matrices
            for l = i+1:Nr
                G_li = eye(Nr);
                psi_li = acos(Omega(i, i)/ sqrt(abs(Omega(i, i))^2 + abs(Omega(l, i))^2));
                
                quantized_psi_li = quantize(psi_li, const1_psi, const2_psi);
                psi_li_rad = inverse_quantize(quantized_psi_li, const1_psi, const2_psi);
                
                G_li(i, i) = cos(psi_li_rad);
                G_li(l, l) = cos(psi_li_rad);
                G_li(i, l) = sin(psi_li_rad);
                G_li(l, i) = -sin(psi_li_rad);
                G_li_matrices{l, i} = G_li;
                Omega = G_li * Omega;
            end
        end   
        
        %% reconstruct V tilde matrix
        I_matrix = eye(Nr, Nc);
        Vtilde = eye(Nr, Nr);
        for i = 1:min(Nc, Nr-1)
            Vtilde = Vtilde * cell2mat(D_i_matrices(i));
            for l = i+1:Nr
                Vtilde = Vtilde * cell2mat(G_li_matrices(l, i)).';
            end
        end
        Vtilde = Vtilde * I_matrix;
        
        Vreconstruct = Vtilde * Dtilde;
        Vreconstruct_matrix(:, s_i, :) = Vreconstruct;
        
    end
    
    V_matrices{k} = V_matrix;
    Vreconstruct_matrices{k} = Vreconstruct_matrix;
    k = k + 1;

end

%% CLEAR UNUSED VARIABLES
clear V_matrix Vreconstruct_matrix Vtildevm_angles_vector S U V Omega 
clear H_matrix G_li_matrices G_li h Vtilde Vreconstruct

%% save 
elements = size(V_matrices, 2);
ant_n = 3;
stream_n = 2;
V_stacked =[V_matrices{:}];
vs = reshape(V_stacked, 3, 234, elements, 2);
name_save_v = strcat('simulation_outputs/v_simulation_psi', num2str(psi_bit), '.mat');
save(name_save_v, 'vs', '-v7.3')

%% CLEAR UNUSED VARIABLES
clear V_matrices V_stacked

%% save 
Vreconstruct_stacked =[Vreconstruct_matrices{:}];
vrs = reshape(Vreconstruct_stacked, 3, 234, elements, 2);
name_save_v_rec = strcat('simulation_outputs/v_reconstructed_simulation_psi', num2str(psi_bit), '.mat');
save(name_save_v_rec, 'vrs', '-v7.3')

%% CLEAR UNUSED VARIABLES
clear Vreconstruct_matrices Vreconstruct_stacked

%% plot
error_quantiz = vs - vrs;

% average on the subcarriers
error_average_subcarriers = squeeze(mean(abs(error_quantiz), 2));

% histograms
if psi_bit == 7
    edges = linspace(2.7e-3, 11.5e-3, 300);
elseif psi_bit == 5
    edges = linspace(1e-2, 4.5e-2, 300);
end
figure(); 
colors = ['r', 'b', 'y', 'c', 'm', 'g'];
i = 1;
for stream_n = 1:2
    for ant_n = 1:3
        error1 = squeeze(error_average_subcarriers(ant_n, :, stream_n));
        histogram(error1, edges, 'FaceColor', colors(i), 'EdgeColor', 'k');
        hold on
        i = i + 1;
    end
end
grid on;
title('Packets error');
legend('V_{1, 1}', 'V_{2, 1}', 'V_{3, 1}', 'V_{1, 2}', 'V_{2, 2}', 'V_{3, 2}');

