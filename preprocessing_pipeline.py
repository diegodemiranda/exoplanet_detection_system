
def preprocess_kepler_data(light_curve_file):
    """
    Pipeline completo de pré-processamento para dados Kepler/TESS
    """
    import numpy as np
    from scipy import signal
    from scipy.interpolate import interp1d

    # 1. Carregamento dos dados
    flux = light_curve_file['PDCSAP_FLUX']  # Flux corrigido pelo PDC
    time = light_curve_file['TIME']
    quality = light_curve_file['SAP_QUALITY']

    # 2. Remoção de pontos com qualidade ruim
    good_quality = quality == 0
    flux_clean = flux[good_quality]
    time_clean = time[good_quality]

    # 3. Remoção de NaNs
    valid_data = ~np.isnan(flux_clean)
    flux_valid = flux_clean[valid_data]
    time_valid = time_clean[valid_data]

    # 4. Detrending usando filtro Savitzky-Golay
    if len(flux_valid) > 99:  # Necessário para window_length
        flux_detrended = signal.savgol_filter(flux_valid, window_length=99, polyorder=3)
        flux_normalized = flux_valid / flux_detrended
    else:
        flux_normalized = flux_valid / np.median(flux_valid)

    # 5. Normalização robusta (median + MAD)
    median_flux = np.median(flux_normalized)
    mad_flux = np.median(np.abs(flux_normalized - median_flux))
    flux_final = (flux_normalized - median_flux) / (1.4826 * mad_flux)

    # 6. Clipagem de outliers
    flux_clipped = np.clip(flux_final, -5, 5)

    # 7. Interpolação para grid uniforme
    if len(time_valid) > 1:
        time_uniform = np.linspace(time_valid[0], time_valid[-1], 2001)
        interp_func = interp1d(time_valid, flux_clipped, 
                              kind='linear', fill_value='extrapolate')
        flux_interpolated = interp_func(time_uniform)
    else:
        flux_interpolated = np.zeros(2001)

    return flux_interpolated, time_uniform

def extract_transit_features(flux, period, epoch, duration):
    """
    Extrai características específicas de trânsito
    """
    features = {}

    # Profundidade do trânsito
    features['transit_depth'] = np.min(flux)

    # Duração relativa
    features['relative_duration'] = duration / period

    # Signal-to-noise ratio
    in_transit = np.abs(flux) > 2 * np.std(flux)
    if np.sum(in_transit) > 0:
        features['snr'] = np.abs(np.mean(flux[in_transit])) / np.std(flux[~in_transit])
    else:
        features['snr'] = 0

    # Assimetria do trânsito
    features['skewness'] = np.mean((flux - np.mean(flux))**3) / (np.std(flux)**3)

    # Curtose
    features['kurtosis'] = np.mean((flux - np.mean(flux))**4) / (np.std(flux)**4)

    return features
