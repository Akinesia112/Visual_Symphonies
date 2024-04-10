from PIL import Image
import os
import math

# Find the closest factors for the number of photos
def find_closest_factors(num_photos):
    sqrt_of_photos = int(math.sqrt(num_photos))
    for factor in range(sqrt_of_photos, 0, -1):
        if num_photos % factor == 0:
            return factor, num_photos // factor
        
# Calculate the size for each image based on the final collage size and grid arrangement        
def calculate_image_size(total_width, total_height, columns, rows):
    return total_width // columns, total_height // rows

# Create the collage
def create_collage(folder_path, output_path, final_collage_size=(9216, 9216)):
    image_paths = [os.path.join(folder_path, img) for img in os.listdir(folder_path) if img.endswith(('jpg', 'jpeg', 'png'))]
    num_photos = len(image_paths)
    if num_photos == 0:
        raise ValueError("No images found in the folder.")

    columns, rows = find_closest_factors(num_photos)
    print(f"Collage will have {columns} columns and {rows} rows.")

    # Calculate the size of each image to fit the final collage size
    new_width, new_height = calculate_image_size(final_collage_size[0], final_collage_size[1], columns, rows)

    # Create a new white background collage
    collage = Image.new('RGBA', final_collage_size, (255, 255, 255, 0))
    
    # Resize each image and paste it into the collage
    for i, img_path in enumerate(image_paths[:columns*rows]):
        with Image.open(img_path) as img:
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            x_position = (i % columns) * new_width
            y_position = (i // columns) * new_height
            collage.paste(img, (x_position, y_position))
    
    # Save the collage
    output_file_path = os.path.join(output_path, f'collage_{num_photos}.png')
    collage.save(output_file_path)
    print(f"Collage image saved at {output_file_path}")
    
create_collage(
    folder_path=r'D:\YTC\ArtPaper\02 - 1500', 
    output_path=r'D:\YTC\ArtPaper\output_img'
    )