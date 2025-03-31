import numpy as np
from PIL import Image


#ETAPA 1 - Leitura de Filtro Normal
def load_filter(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    m, n = map(int, lines[0].split())
    mask = []
    for line in lines[1:]:
        mask.append([float(x) for x in line.strip().split()])
    return np.array(mask), m, n


#ETAPA 2 - Leitura de Filtro com Bias, Stride e Ativação
def load_filter_with_params(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    bias = int(lines[0].split()[1])
    stride = int(lines[1].split()[1])
    activation = lines[2].split()[1]
    m, n = map(int, lines[3].split())
    mask = []
    for line in lines[4:]:
        mask.append([float(x) for x in line.strip().split()])
    return np.array(mask), m, n, bias, stride, activation


#ETAPA 3 - Leitura de Filtro 3D
def load_filter_3d(filepath):
    with open(filepath, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip() != '']
    m, n, d = map(int, lines[0].split())
    mask = []
    index = 1
    for channel in range(d):
        channel_mask = []
        for i in range(m):
            channel_mask.append([float(x) for x in lines[index].strip().split()])
            index += 1
        mask.append(channel_mask)
    return np.array(mask)  # Shape (d, m, n)


#Utilitários
def expand_histogram(img):
    min_val = img.min()
    max_val = img.max()
    return ((img - min_val) * (255.0 / (max_val - min_val))).astype(np.uint8)

def relu(x):
    return np.maximum(0, x)


#ETAPA 1 - Correlação 2D simples
def correlate_channel(channel, kernel):
    kh, kw = kernel.shape
    pad_h = kh // 2
    pad_w = kw // 2
    padded = np.pad(channel, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
    out = np.zeros_like(channel)

    for i in range(channel.shape[0]):
        for j in range(channel.shape[1]):
            region = padded[i:i+kh, j:j+kw]
            out[i, j] = np.sum(region * kernel)
    return out

def correlate_rgb(image, kernel):
    result = np.zeros_like(image)
    for c in range(3):
        result[:, :, c] = correlate_channel(image[:, :, c], kernel)
    return np.clip(result, 0, 255)

def apply_filter(image_path, filter_path, output_path, sobel=False):
    image = np.array(Image.open(image_path).convert('RGB'), dtype=np.float32)
    kernel, _, _ = load_filter(filter_path)
    filtered = correlate_rgb(image, kernel)

    if sobel:
        filtered = np.abs(filtered)
        for c in range(3):
            filtered[:, :, c] = expand_histogram(filtered[:, :, c])

    output_image = Image.fromarray(filtered.astype(np.uint8))
    output_image.save(output_path)


#ETAPA 2 - Correlação com Bias, Stride e Ativação
def correlate_with_params(image, kernel, bias, stride, activation):
    kh, kw = kernel.shape
    H, W, _ = image.shape
    out_h = (H - kh) // stride + 1
    out_w = (W - kw) // stride + 1
    result = np.zeros((out_h, out_w, 3))

    for c in range(3):
        for i in range(out_h):
            for j in range(out_w):
                region = image[i*stride:i*stride+kh, j*stride:j*stride+kw, c]
                val = np.sum(region * kernel) + bias
                if activation == 'relu':
                    val = relu(val)
                result[i, j, c] = val

    return np.clip(result, 0, 255)

def apply_filter_with_params(image_path, filter_path, output_path, sobel=False):
    image = np.array(Image.open(image_path).convert('RGB'), dtype=np.float32)
    kernel, _, _, bias, stride, activation = load_filter_with_params(filter_path)
    filtered = correlate_with_params(image, kernel, bias, stride, activation)

    if sobel:
        filtered = np.abs(filtered)
        for c in range(3):
            filtered[:, :, c] = expand_histogram(filtered[:, :, c])

    output_image = Image.fromarray(filtered.astype(np.uint8))
    output_image.save(output_path)


#ETAPA 3 - Correlação 3D volumétrica
def correlate_3d(image, kernel_3d):
    kh, kw = kernel_3d.shape[1], kernel_3d.shape[2]
    pad_h, pad_w = kh // 2, kw // 2
    H, W, C = image.shape  # Pegamos altura, largura e canais
    output = np.zeros((H, W, C), dtype=np.float32)  # Mantemos os 3 canais
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='reflect')
    
    for c in range(C):  # Processamos cada canal separadamente
        for i in range(H):
            for j in range(W):
                region = padded[i:i+kh, j:j+kw, c]
                output[i, j, c] = np.sum(region * kernel_3d[c])  # Aplicamos o kernel específico do canal
    
    return np.clip(output, 0, 255).astype(np.uint8)

def apply_filter_3d(image_path, filter_path, output_path, sobel=False):
    image = np.array(Image.open(image_path).convert('RGB'), dtype=np.float32)
    kernel_3d = load_filter_3d(filter_path)
    filtered = correlate_3d(image, kernel_3d)

    if sobel:
        filtered = np.abs(filtered)
        filtered = expand_histogram(filtered)

    output_img = np.stack([filtered]*3, axis=-1)
    Image.fromarray(output_img.astype(np.uint8)).save(output_path)


#ETAPA 4 - Conversão para tons de cinza
def replicate_g_channel(image):
    G = image[:, :, 1]
    gray_img = np.stack([G, G, G], axis=-1)
    return gray_img

def convert_to_y_channel(image):
    R = image[:, :, 0]
    G = image[:, :, 1]
    B = image[:, :, 2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    gray_img = np.stack([Y, Y, Y], axis=-1)
    return gray_img

def convert_to_grayscale(image_path, output_g_path, output_y_path):
    image = np.array(Image.open(image_path).convert('RGB'), dtype=np.float32)

    # (a) Replicando G
    gray_g = replicate_g_channel(image)
    Image.fromarray(np.clip(gray_g, 0, 255).astype(np.uint8)).save(output_g_path)

    # (b) Usando Y do YIQ
    gray_y = convert_to_y_channel(image)
    Image.fromarray(np.clip(gray_y, 0, 255).astype(np.uint8)).save(output_y_path)
