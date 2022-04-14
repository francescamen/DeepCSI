function [quantized_angle] = quantize(angle_rad, const1, const2)

quantized_angle = (angle_rad/pi - 1/const2) * const1;

quantized_angle = round(quantized_angle);

end

