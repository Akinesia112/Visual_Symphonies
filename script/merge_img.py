from PIL import Image
import os
import math

# 根据图片总数找到接近正方形的行列比
def find_closest_factors(num_photos):
    sqrt_of_photos = int(math.sqrt(num_photos))
    for factor in range(sqrt_of_photos, 0, -1):
        if num_photos % factor == 0:
            return factor, num_photos // factor

def create_collage(folder_path, output_path):
    image_paths = [os.path.join(folder_path, img) for img in os.listdir(folder_path) if img.endswith(('jpg', 'jpeg', 'png'))]
    
    num_photos = len(image_paths)
    if num_photos == 0:
        raise ValueError("資料夾中沒有找到圖片。")

    n, m = find_closest_factors(num_photos)
    print(f"拼貼將會是 {n} 行和 {m} 列")

    # 打开第一张图片以获取图片的原始尺寸
    with Image.open(image_paths[0]) as img:
        original_width, original_height = img.width, img.height
    
    # 计算拼贴图的总体尺寸
    total_width = original_width * m
    total_height = original_height * n

    # 计算调整后的每张图片尺寸
    new_width = total_width // m
    new_height = total_height // n

    # 创建新的拼贴图,保持去除背景色
    collage = Image.new('RGBA', (total_width, total_height), (255, 255, 255, 255))
    
    # 逐张图片处理
    for i, img_path in enumerate(image_paths[:n*m]):
        with Image.open(img_path) as img:
            # 调整图片尺寸
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # 计算图片在拼贴图中的位置
            x_position = (i % m) * new_width
            y_position = (i // m) * new_height
            collage.paste(img, (x_position, y_position))
    
    collage.save(output_path + f'/collage_{num_photos}.png')
    print(f"拼贴图片已保存到{output_path}")

# 使用示例（请替换路径）
create_collage(
    folder_path=r'D:\YTC\ArtPaper\02 - 1500', 
    output_path=r'D:\YTC\ArtPaper\output_img'
    )
