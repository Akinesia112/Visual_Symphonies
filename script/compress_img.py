import os
from PIL import Image
import os
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

def compress_image(image_path, output_path, quality):
    try:
        with Image.open(image_path) as img:
            # Get the original image size
            original_size = os.path.getsize(image_path)
            
            # Calculate the target size in bytes
            target_size = 25 * 1024 * 1024
            
            # Calculate the maximum number of pixels
            max_pixels = 100000000
            
            # Calculate the compression ratio based on size and pixels
            compression_ratio = min((target_size / original_size) ** 0.5, (max_pixels / (img.width * img.height)) ** 0.5)
            
            # Calculate the new dimensions based on the compression ratio
            new_width = int(img.width * compression_ratio)
            new_height = int(img.height * compression_ratio)
            
            # Resize the image
            resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Save the resized image with the specified quality
            resized_img.save(output_path, "png", quality=quality, optimize=True)
    except Exception as e:
        print(f"Error compressing image: {e}")



# 使用示例
input_path = r"D:\YTC\ArtPaper\output_img\collage_1500.png"
output_path = r"D:\YTC\ArtPaper\output_img\collage_1500_compressed.png"

compress_image(input_path, output_path, quality=20)
