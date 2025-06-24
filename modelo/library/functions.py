import numpy as np

def apply_jittering_xyz(coords, sigma=0.01):
    noise = np.random.normal(loc=0.0, scale=sigma, size=coords.shape)
    return coords + noise

def apply_scaling_xyz(coords, scale_range=(0.9, 1.1)):
    scale = np.random.uniform(*scale_range)
    return coords * scale

def time_warp_sequence(frames, max_shrink=0.1):
    n_real, n_feat = frames.shape
    # elegimos un porcentaje α de 0 a max_shrink
    alpha = np.random.uniform(0, max_shrink)
    new_len = int(np.floor(n_real * (1 - alpha)))
    if new_len < 2:
        # si quedan muy pocos, retornamos tal cual la secuencia original
        return frames.copy()
    
    # Remuestreamos con interpolación lineal: 
    # los índices “tie” entre 0 y n_real-1 para crear new_len pasos.
    orig_indices = np.linspace(0, n_real - 1, n_real)
    new_indices = np.linspace(0, n_real - 1, new_len)
    
    warped = np.zeros((new_len, n_feat), dtype=frames.dtype)
    for j in range(n_feat):
        warped[:, j] = np.interp(new_indices, orig_indices, frames[:, j])
    return warped

def pad_sequence_to_length(frames, target_len):
    n, f = frames.shape
    if n >= target_len:
        return frames[:target_len, :]  # cortamos
    else:
        pad_len = target_len - n
        pad = np.zeros((pad_len, f), dtype=frames.dtype)
        return np.vstack([frames, pad])

def augment_sequence(sequence, target_len=162, sigma=0.01, scale_range=(0.9, 1.1), max_shrink=0.1):
    n_frames, n_feats = sequence.shape
    
    # Crear máscara: True si AL MENOS un valor en ese frame no es cero
    nonzero_mask = np.any(sequence != 0.0, axis=1)
    if not np.any(nonzero_mask):
        # si TODO es padding (caso extremo), retornamos un arreglo de ceros
        return np.zeros((target_len, n_feats), dtype=sequence.dtype)
    
    # Index de últimos “frame real” (=True en nonzero_mask)  
    last_real_idx = np.max(np.where(nonzero_mask)[0])
    real_frames = sequence[: last_real_idx + 1, :].copy()  # copiamos los frames reales
    pad_frames  = sequence[last_real_idx + 1 :, :].copy()  # si existe padding viejo
    
    # 2) Separar coordenadas y visibilidades
    #    Sabemos que cada conjunto de 4 columnas es [x, y, z, v]
    #    Así que hacemos reshape a (n_real, n_landmarks, 4)
    n_landmarks = n_feats // 4
    real_reshaped = real_frames.reshape(-1, n_landmarks, 4)  # shape = (n_real, n_landmarks, 4)
    
    xyz = real_reshaped[:, :, 0:3].copy()   # (n_real, n_landmarks, 3)
    vis = real_reshaped[:, :, 3].copy()     # (n_real, n_landmarks)
    
    # 3) Aplicar jitter y scaling SOLO a (x,y,z)
    #    Recorremos todos los frames y landmarks:
    xyz_jittered = np.zeros_like(xyz)
    for t in range(xyz.shape[0]):
        # a) jitter de ruido gaussiano en coords 3D
        xyz_j = apply_jittering_xyz(xyz[t], sigma=sigma)
        # b) scaling uniforme en coords 3D
        xyz_s = apply_scaling_xyz(xyz_j, scale_range=scale_range)
        xyz_jittered[t] = xyz_s
    
    # 4) Reconstruir frames “completos”: juntamos xyz_jittered con vis intacto
    #    Ahora xyz_jittered tiene forma (n_real, n_landmarks, 3), vis tiene (n_real, n_landmarks)
    real_aug = np.zeros_like(real_reshaped)  # (n_real, n_landmarks, 4)
    real_aug[:, :, 0:3] = xyz_jittered
    real_aug[:, :, 3] = vis
    
    # 5) “Desaplanar” nuevamente a (n_real, n_feats)
    real_aug = real_aug.reshape(-1, n_feats)
    
    # 6) Time-warp SOBRE frames reales (de forma suave con interpolación)
    warped = time_warp_sequence(real_aug, max_shrink=max_shrink)
    
    # 7) Concatenamos cualquier bloque de padding viejo que hubiera (si es que había)
    #    pero ojo: si warped ya excede target_len, lo cortamos.
    if pad_frames.shape[0] > 0:
        full_aug = np.vstack([warped, pad_frames])
    else:
        full_aug = warped
    
    # 8) Finalmente, garantizar longitud EXACTA target_len (cortar o paddear)
    final_seq = pad_sequence_to_length(full_aug, target_len)
    return final_seq
