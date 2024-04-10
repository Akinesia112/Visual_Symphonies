import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import math
from sklearn.cluster import KMeans
import numpy as np
from torchvision import models, transforms
import torch
import os
import random
os.environ["OMP_NUM_THREADS"] = "1"

# 根据图片总数找到接近正方形的行列比
def find_closest_factors(num_photos):
    sqrt_of_photos = int(math.sqrt(num_photos))
    for factor in range(sqrt_of_photos, 0, -1):
        if num_photos % factor == 0:
            return factor, num_photos // factor

def extract_features(image_paths, model, transform):
    features = []
    for img_path in image_paths:
        img = Image.open(img_path).convert("RGB")
        img_t = transform(img)
        batch_t = torch.unsqueeze(img_t, 0)
        out = model(batch_t)
        features.append(out.flatten().detach().numpy())
    return np.array(features)

def calculate_feature_similarity(feature, centroid, max_radius, max_distance):
    # Calculate the Euclidean distance between the feature vector and the centroid
    distance = np.linalg.norm(feature - centroid)
    # Scale the distance to be within a maximum radius
    radius = (distance / max_distance) * max_radius
    return radius

def remove_background(img_path, output_size=(500, 500)):
    with Image.open(img_path) as img:
        # Convert image to RGBA (if not already in this mode)
        img = img.convert("RGBA")
        datas = img.getdata()

        newData = []
        for item in datas:
            # Change all white (also shades of whites)
            # pixels to transparent
            if item[0] > 220 and item[1] > 220 and item[2] > 220:  # Adjust the threshold as necessary
                newData.append((255, 255, 255, 0))
            else:
                newData.append(item)

        img.putdata(newData)
        img = img.resize(output_size, Image.Resampling.LANCZOS)
    return img

def cluster_images(features, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto').fit(features)
    return kmeans.labels_, n_clusters

def create_collage_with_clusters(image_paths, labels, features, output_path, n_clusters):
    # Define the size of the collage based on the number of images and desired output
    collage_width, collage_height = 10000, 10000  # Adjust size as needed

    # Initialize the collage with white background
    collage = Image.new('RGBA', (collage_width, collage_height), (255, 255, 255, 0))

    # Calculate the size of the grid that will hold the clusters
    grid_width = collage_width // n_clusters
    grid_height = collage_height // n_clusters   

    # Assign images to clusters
    clusters = [[] for _ in range(n_clusters)]
    for label, img_path, feature in zip(labels, image_paths, features):
        clusters[label].append((img_path, np.array(feature)))

    # Calculate centroids for non-empty clusters
    centroids = []
    for cluster in clusters:
        if cluster:  # Check if the cluster is not empty
            cluster_features = np.array([feature for _, feature in cluster])
            centroids.append(np.mean(cluster_features, axis=0))

    # Calculate max_distance for non-empty clusters
    max_distance = max(
        np.linalg.norm(features - centroid, axis=1).max()
        for centroid in centroids
        if len(centroid) > 0
    )

    # Calculate max_radius based on the size of the grid that will hold the clusters
    max_radius = min(grid_width, grid_height) // 4  # Adjust the denominator as needed

    # Draw and place images
    draw = ImageDraw.Draw(collage, 'RGBA')
    for i, (cluster, centroid) in enumerate(zip(clusters, centroids)):
        # Calculate the central point for this cluster with radominal distance, but also mutually exclusive with the radius
        r =  max_radius
        while True:
            x = random.randint(0, grid_width)
            y = random.randint(0, grid_height)
            if (x - r) ** 2 + (y - r) ** 2 <= r ** 2:
                break
            r += 1
            
        # Calculate the central point for this cluster
        central_x = int(x + r)
        central_y = int(y + r)
        
        for j, (img_path, feature) in enumerate(cluster):
            # Calculate the placement radius based on feature similarity
            radius = calculate_feature_similarity(feature, centroid, max_radius, max_distance)

            # Determine angle and calculate position
            angle = 2 * math.pi * j / len(cluster)
            x_position = int(central_x + radius * math.cos(angle))
            y_position = int(central_y + radius * math.sin(angle))

            # Remove background, resize, and paste
            img = remove_background(img_path)
            img = img.resize((150, 150))  # Change as needed
            collage.paste(img, (x_position, y_position), img)

    # Save the collage
    collage_path = os.path.join(output_path, f'collage_clustered_{n_clusters}_{len(image_paths)}.png')
    collage.save(collage_path)
    print(f"Collage image has been saved to {collage_path}")

    return collage_path

# 示例使用
folder_path = r'D:\YTC\ArtPaper\01 - 1000'
output_path = r'D:\YTC\ArtPaper\output_img'
image_paths = [os.path.join(folder_path, img) for img in os.listdir(folder_path) if img.endswith(('jpg', 'jpeg', 'png'))]

# 模型和变换
Inceptionv3 = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
Inceptionv3.eval()
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

features = extract_features(image_paths, Inceptionv3, transform)
n_clusters = 5  # 可以根据实际情况调整
labels, n_clusters = cluster_images(features, n_clusters)

num_photos = len(image_paths)
n, m = find_closest_factors(num_photos)
create_collage_with_clusters(image_paths, labels, features, output_path, n_clusters)
