import os
import rasterio
import numpy as np
from PIL import Image

def convert_tif_to_png(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(".tif") or filename.lower().endswith(".tiff"):
            tif_path = os.path.join(input_folder, filename)
            png_filename = os.path.splitext(filename)[0] + ".png"
            png_path = os.path.join(output_folder, png_filename)

            try:
                with rasterio.open(tif_path) as src:
                    # 读取前3个波段（如果存在）
                    count = src.count
                    bands_to_read = min(count, 3)
                    img_array = np.stack([src.read(i + 1) for i in range(bands_to_read)], axis=-1)

                    # 归一化到0-255（避免值超出范围）
                    img_array = img_array.astype(np.float32)
                    img_array -= img_array.min()
                    img_array /= img_array.max()
                    img_array *= 255.0
                    img_array = img_array.astype(np.uint8)

                    # 若波段数不足3，则补充成3通道
                    if img_array.shape[2] == 1:
                        img_array = np.repeat(img_array, 3, axis=2)

                    # 保存为 PNG
                    Image.fromarray(img_array).save(png_path)
                    print(f"已转换: {filename} -> {png_filename}")

            except Exception as e:
                print(f"处理 {filename} 时出错：{e}")


# 示例用法
# input_dir = "dataset/opt_images/train"
# output_dir = "dataset/opt_images/train_png"
# input_dir = "dataset/opt_images/test"
# output_dir = "dataset/opt_images/test_png"
# input_dir = "dataset/sar_images/train"
# output_dir = "dataset/sar_images/train_png"
input_dir = "dataset/sar_images/test"
output_dir = "dataset/sar_images/test_png"
convert_tif_to_png(input_dir, output_dir)
