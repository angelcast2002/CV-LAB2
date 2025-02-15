import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('./rice.jpg', cv2.IMREAD_GRAYSCALE)

umbral, binarizada = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

print(f"Umbral calculado por Otsu: {umbral}")

kernel = np.ones((3, 3), np.uint8)

binarizada_limpia = cv2.morphologyEx(binarizada, cv2.MORPH_OPEN, kernel)

num_labels, labels = cv2.connectedComponents(binarizada_limpia)

print(f"Número de granos de arroz: {num_labels - 1}")

fig, axs = plt.subplots(1, 3, figsize=(12, 4))
axs[0].imshow(img, cmap='gray')
axs[0].set_title('Original (grises)')
axs[0].axis('off')

axs[1].imshow(binarizada, cmap='gray')
axs[1].set_title('Binarizada (Otsu)')
axs[1].axis('off')

axs[2].imshow(binarizada_limpia, cmap='gray')
axs[2].set_title('Tras apertura morfológica')
axs[2].axis('off')

plt.tight_layout()
plt.show()