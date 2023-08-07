%% Code for generating a B-mode image of the first frame

%raw data path
data_path = '/local_disk/Datasets/PlaneWaveImaging/20230118/Eval/IQdata/';
%save path
save_root_path = '/home/jaxa/Datasets/PlaneWaveImaging/20230118/Breastfan/IQdata/';

sub_dir = dir(strcat(data_path, '*.mat'));
[sub_num, temp] = size(sub_dir);

for i= 1:sub_num
    clear IQData;
    clear I_Data;
    clear Q_Data;
    mat_path = strcat(data_path, sub_dir(i).name)
    load(mat_path);
    IQData = squeeze(complex(I_Data, Q_Data));
    % save_path = strcat(data_path, sprintf('%04d', i));
    save_path = strcat(save_root_path, sprintf('%04d', i));
    mkdir(save_path);
    [temp1, temp2, scan_num, frame_num] = size(IQData);
    for j= 1:frame_num
        frame_save_path = strcat(save_path, '/', sprintf('%04d', j));
        %mkdir(frame_save_path);
        %mkdir(strcat(frame_save_path, '/rf_real'));
        %mkdir(strcat(frame_save_path, '/rf_imag'));

        frame_data = IQData(:, :, :, j);

        comp_IQData = sum(frame_data, 3);
        comp_envelope = abs(comp_IQData) ./ max(abs(comp_IQData(:)));
        comp_rf_real = real(comp_IQData);
        comp_rf_imag = imag(comp_IQData);
        comp_Bmode = convert_rf(sum(frame_data, 3), -60);
        
        figure;imagesc(comp_Bmode);
        defaultstyle
        caxis([-60 0])
        saveas(gcf, 'comp.eps');
        keyboard
        saveas(gcf, strcat(frame_save_path, '/', 'comp.png'));
        close all;

        for k= 1:75
            input = convert_rf(frame_data(:, :, k), -60);
            filename = sprintf('%03d.png', k);
            figure;imagesc(input);colormap gray;
            axis off
            saveas(gcf, strcat(frame_save_path, '/', filename));
            close all;
        end

        keyboard

        input = convert_rf(frame_data(:, :, 1), -60);
        figure;imagesc(input);colormap gray;
        saveas(gcf, strcat(frame_save_path, '/', 'input1.png'));
        close all;

        input = convert_rf(frame_data(:, :, 75), -60);
        figure;imagesc(input);colormap gray;
        saveas(gcf, strcat(frame_save_path, '/', 'input75.png'));
        close all;

        
        save(strcat(frame_save_path, '/rf_real/', 'comp_rf_real.mat'), 'comp_rf_real');
        save(strcat(frame_save_path, '/rf_imag/', 'comp_rf_imag.mat'), 'comp_rf_imag');

        rf = frame_data(:, :, 38);
        rf_real = real(frame_data(:, :, 38));
        rf_imag = imag(frame_data(:, :, 38));
        % envelope = abs(rf) ./ max(abs(rf(:)));
        % Bmode = convert_rf(rf, -60);
        filename = sprintf('%04d', 38);
        save(strcat(frame_save_path, '/rf_real/', filename, '.mat'), 'rf_real');
        save(strcat(frame_save_path, '/rf_imag/', filename, '.mat'), 'rf_imag');

        rf = frame_data(:, :, 1);
        rf_real = real(frame_data(:, :, 1));
        rf_imag = imag(frame_data(:, :, 1));
        % envelope = abs(rf) ./ max(abs(rf(:)));
        % Bmode = convert_rf(rf, -60);
        filename = sprintf('%04d', 1);
        save(strcat(frame_save_path, '/rf_real/', filename, '.mat'), 'rf_real');
        save(strcat(frame_save_path, '/rf_imag/', filename, '.mat'), 'rf_imag');

        rf = frame_data(:, :, 75);
        rf_real = real(frame_data(:, :, 75));
        rf_imag = imag(frame_data(:, :, 75));
        % envelope = abs(rf) ./ max(abs(rf(:)));
        % Bmode = convert_rf(rf, -60);
        filename = sprintf('%04d', 75);
        save(strcat(frame_save_path, '/rf_real/', filename, '.mat'), 'rf_real');
        save(strcat(frame_save_path, '/rf_imag/', filename, '.mat'), 'rf_imag');

    end
end