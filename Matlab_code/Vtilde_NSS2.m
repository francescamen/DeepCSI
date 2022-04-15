
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

function vtilde_matrix = Vtilde_NSS2(beamforming_angles, Nc, Nr, NSUBC_VALID, phi_bit, psi_bit)
 %% convert quantized values to angles
const1_phi = 1/2^(phi_bit-1);
const2_phi = 1/2^(phi_bit);
phi_11 = pi * (const2_phi + const1_phi * beamforming_angles(:, 1));
phi_21 = pi * (const2_phi + const1_phi * beamforming_angles(:, 2));
phi_22 = pi * (const2_phi + const1_phi * beamforming_angles(:, 5));

const1_psi = 1/2^(psi_bit+1);
const2_psi = 1/2^(psi_bit+2);
psi_21 = pi * (const2_psi + const1_psi * beamforming_angles(:, 3));
psi_31 = pi * (const2_psi + const1_psi * beamforming_angles(:, 4));
psi_32 = pi * (const2_psi + const1_psi * beamforming_angles(:, 6));

vtilde_matrix = zeros(Nc, NSUBC_VALID, Nr);
for s_i=1:NSUBC_VALID
    %% build D matrices (phi)
    D_1 = [exp(1i*phi_11(s_i)), 0, 0; 
           0, exp(1i*phi_21(s_i)), 0;
           0, 0, 1];
    D_2 = [1, 0, 0; 
           0, exp(1i*phi_22(s_i)), 0;
           0, 0, 1];

    %% build G matrices (psi)
    G_21 = [cos(psi_21(s_i)), sin(psi_21(s_i)), 0; 
            -sin(psi_21(s_i)), cos(psi_21(s_i)), 0;
            0, 0, 1];
    G_31 = [cos(psi_31(s_i)), 0, sin(psi_31(s_i)); 
            0, 1, 0;
            -sin(psi_31(s_i)), 0, cos(psi_31(s_i))];
    G_32 = [1, 0, 0; 
            0, cos(psi_32(s_i)), sin(psi_32(s_i));
            0, -sin(psi_32(s_i)), cos(psi_32(s_i))];

    %% reconstruct V tilde matrix
    I_matrix = eye(Nr, Nc);
    Vtilde = D_1 * G_21.' * G_31.' * D_2 * G_32.' * I_matrix;
    vtilde_matrix(:, s_i, :) = Vtilde.';

end