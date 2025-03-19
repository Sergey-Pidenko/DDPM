import gc
from tqdm import tqdm
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

def denoise_image(model, cond_model, low_res, noisy_image, betas, alphas, alpha_cumprod, num_steps):
    """
    Применяет диффузионную модель для удаления шума из изображения.

    :param model: Обученная диффузионная модель
    :param noisy_image: Изображение с шумом (входное изображение)
    :param betas: Параметры диффузии
    :param num_steps: Количество шагов для процесса восстановления
    :return: Расшумленное изображение
    """
    # Определяем устройство
    device = noisy_image.device

    # Получаем условие из cond_model
    cond, _, _ = cond_model(low_res.to(device))

    model.eval()
    with torch.no_grad():
        # Начальное изображение
        image = noisy_image.clone().to(device)

        # Обратный диффузионный процесс
        for step in tqdm(range(num_steps - 1, -1, -1)):
            t = torch.full((image.size(0),), step, dtype=torch.long, device=device).to(device)

            # Формируем вход для модели: [изображение, условие, временной признак]
            inp = torch.cat([image, cond], dim=1)

            # Прогноз модели о шуме
            predicted_noise = model(inp, t)

            # Обратный шаг диффузии - обновление изображения
            if step > 0:
                noise = torch.randn_like(image)  # добавляем шум кроме самого последнего шага
            else:
                noise = torch.zeros_like(image)  # на последнем шаге нет добавленного шума

            # Из формул для обратного процесса:
            beta_t = betas[step]
            alpha_t = alpha_cumprod[step]

            # “Шаг назад” по формуле
            image = (1 / torch.sqrt(alpha_t)) * (
                image - (beta_t / torch.sqrt(1 - alpha_t)) * predicted_noise
            ) + torch.sqrt(beta_t) * noise
        
    return image


def train(model, cond_model, train_loader, optimizer, loss_fn, device, betas, T=1000, batch_size=8):
    model.train()
    for low_res, high_res in train_loader:
        t = torch.randint(low=0, high=T, size=(batch_size,), dtype=torch.long)
        noisy_images, noise = q_sample(high_res, t, betas)

        low_res, high_res = low_res.to(device), high_res.to(device)
        noisy_images, noise = noisy_images.to(device), noise.to(device)
        
        cond, _, _ = cond_model(low_res)

        inp = torch.concat([noisy_images, cond], dim=1)
        t = t.to(device)
        optimizer.zero_grad()
        # Прямой проход
        outputs = model(inp, t)
        # Вычисление потери
        loss = loss_fn(outputs, noise)
        # Назад и оптимизация
        loss.backward()
        optimizer.step()

    torch.cuda.empty_cache()
    gc.collect()

def validate(model, cond_model, val_loader, loss_fn, device, betas, T=1000, batch_size=8):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for low_res, high_res in val_loader:
            t = torch.randint(low=0, high=T, size=(batch_size,), dtype=torch.long)
            noisy_images, noise = q_sample(high_res, t, betas)

            low_res, high_res = low_res.to(device), high_res.to(device)
            noisy_images, noise = noisy_images.to(device), noise.to(device)

            cond, _, _ = cond_model(low_res)

            inp = torch.concat([noisy_images, cond], dim=1)
            t = t.to(device)
            # Прямой проход
            outputs = model(inp, t)
            # Вычисление потери
            val_loss += loss_fn(outputs, noise).item()

        torch.cuda.empty_cache()
        gc.collect()

    return val_loss / len(val_loader)
