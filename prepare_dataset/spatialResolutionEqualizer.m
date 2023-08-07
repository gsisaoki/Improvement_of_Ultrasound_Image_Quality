function [output] = spatialResolutionEqualizer(image1, res_u, res_v)
    % res_u, res_v：　画像の空間解像度 例えば 0.3, 0.0507
    %Assumes res_u < res_v
    [size_v,size_u] = size(image1);
    u_coords = 1:size_u; % 0, 0.3, 0.6, 0.9,... 
    v_coords = 1:size_v; % 0, 0.0507, 0.1014,...
    [u0,v0] = meshgrid(u_coords, v_coords);  %入力画像の空間情報（座標）
    interp_coords = 1:(res_u / res_v):(size_v+5); %画像サイズの調整のため+5を適用
    [u,v] = meshgrid(u_coords, interp_coords);    %画素値を知りたい座標
    output = interp2(u0, v0, image1, u, v,'linear');
end
