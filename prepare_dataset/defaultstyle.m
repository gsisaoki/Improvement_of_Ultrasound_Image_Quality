colormap gray

Fc = 7.6; % in MHz
speedOfSound = 1540; % in m/s
wavelength = speedOfSound / Fc / 1e3; % in mm

delta_x = 0.5; %in wavelength
delta_z = 0.25; %in wavelength

offset_x = -94.0130; % the position of the first element in wavelenth
offset_z = 5; % startDepth in wavelength


c = colorbar;
c.Color = [1 1 1];
c.Label.String = 'Intensity [dB]';
c.Direction = 'reverse';
c.Location = 'southoutside';

set(gcf,'color',[0,0,0]);
set(gca,'XColor',[1 1 1]);
set(gca,'YColor',[1 1 1]);

xtickvalues = [-15 -10 -5 0 5 10 15]; %in mm
a = delta_x * wavelength;
b = wavelength*(offset_x-delta_x);
xticks((xtickvalues-b)/a)
xticklabels(xtickvalues)
xlabel('Lateral position [mm]')

ytickvalues = [5 10 15 20 25 30 35 40 45 50 55 60]; %in mm
b = wavelength * offset_z - delta_z;
a = delta_z * wavelength;
yticks((ytickvalues-b)/a)
yticklabels(ytickvalues)
ylabel('Depth [mm]')