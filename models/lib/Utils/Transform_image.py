import torch
import torch.nn.functional as F

def resize_image_torch(image):
    """Resize (C, 33, 33) to (C, 224, 224) using bilinear interpolation"""
    image = torch.from_numpy(image).unsqueeze(0)  # Add batch dimension → (1, C, 33, 33)
    resized_image = F.interpolate(image, size=(224, 224), mode='bilinear', align_corners=False)
    return resized_image.squeeze(0)  # Remove batch dimension → (C, 224, 224)

# Example usage:
# Assuming `image` is a numpy array of shape (C, 33, 33)
# resized_image = resize_image_torch(image)
# Note: Make sure to import numpy if you're using a numpy array.