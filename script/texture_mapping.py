import os
import open3d as o3d
from PIL import Image
import numpy as np

# Define the file paths
# model_path = r'D:\YTC\ArtPaper\3D_model_obj\20180310_KickAir8P_UVUnwrapped_Stanford_Bunny_OBJ-JPG\20180310_KickAir8P_UVUnwrapped_Stanford_Bunny.obj'
# mtl_path = r'D:\YTC\ArtPaper\3D_model_obj\20180310_KickAir8P_UVUnwrapped_Stanford_Bunny_OBJ-JPG\20180310_KickAir8P_UVUnwrapped_Ceramic_Stanford_Bunny.mtl'

model_path = r'D:\YTC\ArtPaper\3D_model_obj\20180310_KickAir8P_UVUnwrapped_Stanford_Bunny_OBJ-JPG\samurai_monastry (1).obj'
mtl_path = r'D:\YTC\ArtPaper\3D_model_obj\20180310_KickAir8P_UVUnwrapped_Stanford_Bunny_OBJ-JPG\samurai_monastry (1).mtl'
texture_terracotta_path = r'D:\YTC\ArtPaper\output_img\collage_1500.png'
image_path_new = r'D:\YTC\ArtPaper\output_img\collage_1500_compressed_new.png'
texture_ceramic_path = r'D:\YTC\ArtPaper\3D_model_obj\20180310_KickAir8P_UVUnwrapped_Stanford_Bunny_OBJ-JPG\bunnystanford_res1_UVmapping3072_g005c.jpg'

texture_eye_path = r'D:\YTC\ArtPaper\3D_model_obj\20180310_KickAir8P_UVUnwrapped_Stanford_Bunny_OBJ-JPG\BunnyEye001_g003.png'

save_path_with_texture = r'D:\YTC\ArtPaper\3D_model_obj\20180310_KickAir8P_UVUnwrapped_Stanford_Bunny_OBJ-JPG\samurai_monastry (1)_textured.obj'

def resize_image(image_path, max_size):
    """Resize image and change background to white."""
    with Image.open(image_path) as img:
        # Calculate the new size keeping the aspect ratio
        ratio = min(max_size / img.size[0], max_size / img.size[1])
        new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
        
        # Resize the image
        resized_img = img.resize(new_size, Image.Resampling.LANCZOS)
        
        # Create a new image with a white background
        new_img_with_bg = Image.new('RGB', new_size, (255, 255, 255))
        
        # If the original image has alpha channel, we use it as a mask for pasting
        if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
            # Paste the resized image onto the white background using the alpha channel as mask
            new_img_with_bg.paste(resized_img, (0, 0), resized_img.split()[3])
        else:
            new_img_with_bg.paste(resized_img, (0, 0))
        
        # Save the new image with white background
        new_img_with_bg.save(image_path_new)
        
resize_image(texture_terracotta_path, max_size=3072)

# Load the mesh
mesh = o3d.io.read_triangle_mesh(model_path)
if not mesh.has_vertex_normals():
    mesh.compute_vertex_normals()

# Load the textures
texture_ceramic = o3d.io.read_image(texture_ceramic_path)
texture_terracotta = o3d.io.read_image(image_path_new)
texture_eye = o3d.io.read_image(texture_eye_path)


# Since the model should already have UVs from the .obj, we don't need to assign them manually
# We just need to add the texture to the mesh
mesh.textures = [texture_ceramic, texture_terracotta, texture_eye]

# Save the textured model for later visualization
o3d.io.write_triangle_mesh(save_path_with_texture, mesh)

# Visualize the mesh with textures
o3d.visualization.draw_geometries([mesh], window_name='Textured Bunny')