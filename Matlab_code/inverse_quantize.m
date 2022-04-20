
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

function [angle_rad] = inverse_quantize(quantized_angle, const1, const2)

angle_rad = pi * (1/const2 + quantized_angle/const1);

end

