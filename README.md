# Traceability Analysis of Image Protection Methods

## Overview

This project studies the robustness performance of adversarial-based and watermark-based protection methods (Glaze, Mist, PhotoGuard, VINE) under series of image transformations.

---
## Structure and Function
#### 1. **utils.py**  - includes 10 transformation methods
| Method | Function | Detail |
|--------|------|---------|
| `crop_and_resize` | adjust the image size | crop 128 pixels' edge，resize to 512x512 |
| `gaus_blur` | Gaussian blurring | kernel: 3x3，sigma=0.05 |
| `jpeg_compress` | JPEG compression | factor=75 |
| `horizontal_flip` | horizontal flipping | cv2.flip |
| `brightness_contrast` | adjust the brightness and contrast | brightness=10, contrast=2 |
| `super_resolution` | Super resolution | EDSR model |
| `distortion` | barrier distortion | k=-0.5 |
| `realistic` | AI-base stylization | SDXL "Transform to realistic photograph" |
| `style` | AI-based stylization | SDXL "Change to Ghibli style" |
| `semantic` | AI-based semantic editing | Gemini detects the object list, SDXL remove certain objects from the image |

2. **apply.py** - script that applys the transformation 
```
Level 1: single transformation
  e.g.: semantic, style, realistic, etc.

Level 2: combination of two non-repeatable transformations
  e.g.: semantic_style, realistic_crop_and_resize
  #: 10 × 9 = 90 types

Level 3: combination of three non-repeatable transformations
  #: 50 types (10 × 5)

Level 4: combination of four non-repeatable transformations
  #: 30 types (10 × 3)

Level 5: combination of five non-repeatable transformations
  #: 20 types (10 × 2)
```
3. **eval/psnr.py** - evaluate Peak Signal-to-Noise Ratio quality
```python
PSNR = 20 × log10(MAX / √MSE)
```