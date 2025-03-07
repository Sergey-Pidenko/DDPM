import torch

def q_sample(x_start, t, betas):
    """
    Генерирует зашумленный образец `x_t` из `x_start` на шаге `t`.
    
    :param x_start: Исходное изображение
    :param t: Временной шаг
    :param betas: Параметры диффузионного процесса (увеличение шума на каждом шаге)
    :return: x_t, зашумленный образец
    """
    noise = torch.randn_like(x_start)
    cumprod_betas = torch.cumprod(1 - betas, dim=0)
    
    sqrt_alphas_cumprod_t = torch.sqrt(cumprod_betas[t]).view(-1, 1, 1, 1)
    sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1 - cumprod_betas[t]).view(-1, 1, 1, 1)

    noisy_image = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    return noisy_image, noise
