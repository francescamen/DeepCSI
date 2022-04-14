function [angle_rad] = inverse_quantize(quantized_angle, const1, const2)

angle_rad = pi * (1/const2 + quantized_angle/const1);

end

