import os
import cv2
import numpy as np

# Diretório raiz das imagens originais
input_root = "C:\\Users\\RianA\\Documents\\GitHub\\the-dip\\processed_images"
output_root = "C:\\Users\\RianA\\Documents\\GitHub\\the-dip\\augmented_dataset"

os.makedirs(output_root, exist_ok=True)

for class_name in os.listdir(input_root):
    class_path = os.path.join(input_root, class_name)

    if not os.path.isdir(class_path):
        continue  # Pula arquivos que não são diretórios

    output_class_path = os.path.join(output_root, class_name)
    os.makedirs(output_class_path, exist_ok=True)

    image_files = [f for f in os.listdir(class_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

    for img_name in image_files:
        img_path = os.path.join(class_path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"Erro ao carregar a imagem: {img_path}")
            continue

        base_name, ext = os.path.splitext(img_name)

        # Aplicar Logaritmo
        log_img = np.log1p(img.astype(np.float32) + 1e-5)  # Evita valores zero
        log_img = cv2.normalize(log_img, None, 0, 255, cv2.NORM_MINMAX)
        log_img = np.nan_to_num(log_img)
        log_img = np.uint8(np.clip(log_img, 0, 255))

        # Aplicar Exponencial
        exp_img = np.exp(img.astype(np.float32) / 255.0) - 1
        exp_img = cv2.normalize(exp_img, None, 0, 255, cv2.NORM_MINMAX)
        exp_img = np.nan_to_num(exp_img)
        exp_img = np.uint8(np.clip(exp_img, 0, 255))

        # Aplicar Filtro da Média (Convolução)
        kernel = np.ones((5, 5), np.float32) / 25
        mean_filtered_img = cv2.filter2D(img, -1, kernel)
        mean_filtered_img = np.clip(mean_filtered_img, 0, 255)
        mean_filtered_img = np.uint8(mean_filtered_img)

        # Garantir que os arquivos serão salvos corretamente
        try:
            cv2.imwrite(os.path.join(output_class_path, f"{base_name}-log{ext}"), log_img)
            cv2.imwrite(os.path.join(output_class_path, f"{base_name}-exp{ext}"), exp_img)
            cv2.imwrite(os.path.join(output_class_path, f"{base_name}-mean{ext}"), mean_filtered_img)
        except Exception as e:
            print(f"Erro ao salvar imagens aumentadas para {img_name}: {e}")

print("Aumento de dataset concluído com sucesso!")
