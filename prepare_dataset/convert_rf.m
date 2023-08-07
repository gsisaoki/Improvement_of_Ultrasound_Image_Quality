function [Bmode] = convert_rf(RFdata, threshold)
    Envelope = abs(RFdata)./max(abs(RFdata(:)));
    Bmode = 20 * log10(Envelope);
    Bmode(Bmode < threshold) = threshold;
    % Bmode = spatialResolutionEqualizer(Bmode, 0.3, 0.3);
    % Bmode = spatialResolutionEqualizer(Bmode, 0.3, 0.0507 * 2);
    Bmode(isnan(Bmode)) = threshold;