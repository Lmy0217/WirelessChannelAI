close all
clear all

M = csvread('train.csv');

d = sqrt((M(:, 2) - M(:, 13)) .^ 2 + (M(:, 3) - M(:, 14)) .^ 2) ./ 1000;
pl = M(:, 9) - (46.3 + 33.9 * log10(M(:, 8)) - 13.82 * log10(M(:, 4) + M(:, 10) + 1) + (44.9 - 6.55 * log10(M(:, 16) + M(:, 15) + 1) - 30) .* log10(d + 1) - 15);

hb = M(:, 4) + M(:, 10) - M(:, 15);
s = sqrt(hb .^ 2 + (d * 1000) .^ 2);
hv = M(:, 4) + M(:, 10) - M(:, 15) - d * 1000 .* tan((M(:, 6) + M(:, 7)) .* pi ./ 180);

nx = M(:, 13) - M(:, 2);
ny = M(:, 14) - M(:, 3);

a = zeros(size(nx));
a(ny == 0 & nx > 0) = 90;
a(ny == 0 & nx < 0) = 270;
idxa = ny > 0;
a(idxa) = atan(nx(idxa) ./ ny(idxa)) / pi * 180;
idxa = ny < 0;
a(idxa) = 180 - atan(nx(idxa) ./ ny(idxa)) / pi * 180;
a = abs(a - M(:, 5));
idxa = a >= 360;
a(idxa) = a(idxa) - 360;
idxa = a > 180;
a(idxa) = 360 - a(idxa);

nx = nx / 5;
ny = ny / 5;

x = [nx, ny, M(:, 4), M(:, 5), a, M(:, 6), M(:, 7), M(:, 8), M(:, 9), M(:, 10), M(:, 11), M(:, 12), d, hv, M(:, 15), M(:, 16), M(:, 17), pl];
y = M(:, 18);
save('x.mat', 'x');
save('y.mat', 'y');
