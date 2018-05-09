
n = 101;
x = linspace(-3, 3, n);
y = linspace(-3, 3, n);
z = linspace(-3, 3, n);

% simple
P1 = [0,0,0];
P2 = [1,0,0];
P3 = [0,1,0];
slip = [1.0, 0, 0];
filename = 'result_simple.mat';

% complex
P1 = [0,0.1,0.1];
P2 = [1,-0.2,-0.2];
P3 = [1,1,0.3];
slip = [1.3,1.4,1.5];
filename = 'result_complex.mat';

[X,Y,Z] = meshgrid(x, y, z);
Xf = reshape(X, numel(X), 1);
Yf = reshape(Y, numel(Y), 1);
Zf = reshape(Z, numel(Z), 1);
% Xf, Yf, Zf


% 1 million TDE evaluations takes 1 second on 1 core on meade03
tic;
[UEf, UNf, UVf] = TDdispFS(Xf, Yf, Zf, P1, P2, P3, slip(1), slip(2), slip(3), 0.25);
toc;
tic;
[Stress, Strain] = TDstressFS(Xf, Yf, Zf, P1, P2, P3, slip(1), slip(2), slip(3), 1.0, 1.0);
toc;
format long;
save(filename, 'UEf', 'UNf', 'UVf', 'Stress', 'Strain');


% UEf(isnan(UEf)) = 0;
% UNf(isnan(UEf)) = 0;
% UVf(isnan(UEf)) = 0;
% 
% UE = reshape(UEf, n, n, n);
% 
% f = figure('visible','off');
% contourf(y, z, reshape(UE(:, (n - 1) / 2 + 1, :), n, n));
% xlabel('x');
% ylabel('y');
% colorbar();
% saveas(f, 'figure.png');
