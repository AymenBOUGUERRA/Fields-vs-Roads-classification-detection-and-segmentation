# Import necessary libraries
import os
import torch
import argparse
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import csv
import random
import matplotlib.pyplot as plt

# Function to load a model
def load_model(model_path, num_classes):
    """
    Load a trained model from a checkpoint and modify its architecture.

    Args:
        model_path (str): Path to the saved model file.
        num_classes (int): Number of classes in the classification problem.

    Returns:
        nn.Module: The loaded and modified model.
    """
    model = models.resnet50(pretrained=False)  # Load ResNet-50 architecture without pre-trained weights
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 256),
        nn.ReLU(),
        nn.Dropout(0.45),  # Adding dropout
        nn.Linear(256, num_classes)
    )

    checkpoint = torch.load(model_path)  # Load the model state_dict from the checkpoint
    model.load_state_dict(checkpoint['model_state_dict'])  # Load the trained weights into the model
    model.eval()  # Set the model to evaluation mode
    return model

# Function to perform inference on images using the model
def perform_inference(model, image_paths, class_names, output_csv):
    """
    Perform inference on a list of images using the provided model.

    Args:
        model (nn.Module): The loaded model for inference.
        image_paths (list): List of paths to the images to be inferred.
        class_names (list): List of class names for labeling predictions.
        output_csv (str): Path to the output CSV file to save inference results.

    Returns:
        None
    """
    predictions = []

    with open(output_csv, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Image', 'Predicted Class'])

        for image_path in image_paths:
            image = Image.open(image_path).convert("RGB")  # Open the image and convert to RGB format
            image = data_transforms(image).unsqueeze(0).to(device)  # Apply data transforms and move to device

            with torch.no_grad():
                output = model(image)  # Perform inference
                _, pred = torch.max(output, 1)  # Get the index of the predicted class

            predicted_class = class_names[pred.item()]  # Get the class name from class_names list
            predictions.append((os.path.basename(image_path), predicted_class))
            csv_writer.writerow([os.path.basename(image_path), predicted_class])

            print(f"Image: {image_path} | Predicted Class: {predicted_class}")

    print(f'Inference completed. Results saved in {output_csv}')

# Function to display randomly selected images with predictions
def print_random_images(image_paths, class_names, model, num_images=10):
    """
    Display randomly selected images along with predicted classes.

    Args:
        image_paths (list): List of paths to the images to be displayed.
        class_names (list): List of class names for labeling predictions.
        model (nn.Module): The loaded model for inference.
        num_images (int): Number of random images to display.

    Returns:
        None
    """
    random.shuffle(image_paths)
    print("\nRandomly selected images:")

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle("Randomly Selected Images")

    for i in range(num_images):
        image_path = image_paths[i]
        image = Image.open(image_path)
        image = data_transforms(image).unsqueeze(0).to(device)  # Apply data transforms and move to device

        with torch.no_grad():
            output = model(image)  # Perform inference
            _, pred = torch.max(output, 1)  # Get the index of the predicted class

        predicted_class = class_names[pred.item()]  # Get the class name from class_names list
        image_name = os.path.basename(image_path)  # Extract the image name

        print(f"Image {i + 1}: {image_name} | Predicted Class: {predicted_class}")

        row = i // 5
        col = i % 5
        ax = axes[row, col]

        # Convert the tensor image to a numpy array and adjust the values to [0, 1]
        image_numpy = image.squeeze().cpu().permute(1, 2, 0).numpy()
        image_numpy = (image_numpy - image_numpy.min()) / (image_numpy.max() - image_numpy.min())

        ax.imshow(image_numpy)  # Display the adjusted image
        ax.set_title(f"{image_name}\nPredicted Class: {predicted_class}")  # Display image name and predicted class
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("random_images.png")  # Save the figure as an image file
    plt.show()
    print("Images saved as random_images.png in the root directory")

# Main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform inference on test images using a saved model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the saved model file")
    parser.add_argument("--test_image_dir", type=str, required=True, help="Path to the directory containing test images")
    parser.add_argument("--output_csv", type=str, default="inference_results.csv", help="Path to the output CSV file")
    parser.add_argument("--gpu", action="store_true", help="Use GPU if available")
    args = parser.parse_args()

    class_names = ['fields', 'roads']  # Update with your class names
    num_classes = len(class_names)

    data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    image_paths = [os.path.join(args.test_image_dir, filename) for filename in os.listdir(args.test_image_dir)]

    device = torch.device("cuda:0" if args.gpu and torch.cuda.is_available() else "cpu")
    model = load_model(args.model_path, num_classes).to(device)

    perform_inference(model, image_paths, class_names, args.output_csv)
    print_random_images(image_paths, class_names, model)
