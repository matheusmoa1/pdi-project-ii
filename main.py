from utils import (
    apply_filter,
    apply_filter_with_params,
    apply_filter_3d,
    convert_to_grayscale
)

# IMAGE_PATH = 'images/Shapes.png'
IMAGE_PATH = 'images/testpat.1k.color2.tif'


#ETAPA 1 - Correlação Simples
# print("Executando ETAPA 1 - Correlação Simples...")

# apply_filter(IMAGE_PATH, 'filters/gaussian5x5.txt', 'output/gaussian5x5.png')
# apply_filter(IMAGE_PATH, 'filters/box1x10.txt', 'output/box1x10.png')
# apply_filter(IMAGE_PATH, 'filters/box10x1.txt', 'output/box10x1.png')
# apply_filter(IMAGE_PATH, 'filters/box10x10.txt', 'output/box10x10.png')
# apply_filter(IMAGE_PATH, 'filters/sobel_horizontal.txt', 'output/sobel_horizontal.png', sobel=True)
# apply_filter(IMAGE_PATH, 'filters/sobel_vertical.txt', 'output/sobel_vertical.png', sobel=True)

# print("ETAPA 1 finalizada - Imagens salvas na pasta output/")


#ETAPA 2 - Correlação com BIAS, STRIDE e ReLU
# print("Executando ETAPA 2 - Correlação com Bias, Stride e ReLU...")

# apply_filter_with_params(IMAGE_PATH, 'filters/gaussian_bias.txt', 'output/gaussian_bias.png')
# apply_filter_with_params(IMAGE_PATH, 'filters/sobel_bias.txt', 'output/sobel_bias.png', sobel=True)

# print("ETAPA 2 finalizada - Imagens salvas na pasta output/")


#ETAPA 3 - Correlação 3D Volumétrica
print("Executando ETAPA 3 - Correlação 3D...")

apply_filter_3d(IMAGE_PATH, 'filters/box5x5x3.txt', 'output/box5x5x3.png')
apply_filter_3d(IMAGE_PATH, 'filters/sobel3x3x3.txt', 'output/sobel3x3x3.png', sobel=True)

print("ETAPA 3 finalizada - Imagens salvas na pasta output/")


#ETAPA 4 - Conversão para Tons de Cinza
# print("Executando ETAPA 4 - Conversão para Tons de Cinza...")

# convert_to_grayscale(
#     IMAGE_PATH,
#     'output/gray_g.png',   # (a) Replicando a banda G
#     'output/gray_y.png'    # (b) Usando Y do YIQ
# )

# print("ETAPA 4 finalizada - Imagens salvas na pasta output/")
