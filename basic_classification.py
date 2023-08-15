# Import necessary libraries
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.datasets as datasets
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import math


# Function to calculate cosine annealing learning rate
def cosine_annealing_lr(current_epoch, num_epochs, lr0, lrf):
    """
    Calculate the learning rate using the cosine annealing schedule.

    Args:
        current_epoch (int): The current epoch number.
        num_epochs (int): Total number of epochs.
        lr0 (float): Initial learning rate.
        lrf (float): Final learning rate.

    Returns:
        float: Calculated learning rate.
    """
    return lrf + 0.5 * (lr0 - lrf) * (1 + math.cos(math.pi * current_epoch / num_epochs))


# Function to train the model
def train_model(model, dataloaders, criterion, optimizer, num_epochs, device, writer, model_dir, experiment_name):
    """
    Train the model using the provided data loaders and hyperparameters.

    Args:
        model (nn.Module): The neural network model.
        dataloaders (dict): Dictionary containing data loaders for training and validation.
        criterion (nn.Module): Loss function for training.
        optimizer (optim.Optimizer): Optimization algorithm.
        num_epochs (int): Number of training epochs.
        device (torch.device): Device to run the training on (cuda or cpu).
        writer (SummaryWriter): TensorBoard writer for logging.
        model_dir (str): Directory to save trained models.
        experiment_name (str): Name of the experiment.

    Returns:
        None
    """
    best_acc = 0.0
    best_loss = float('inf')  # Initialize with a large value
    global_step = 0

    # Loop through each epoch
    for epoch in range(num_epochs):
        # Calculate the learning rate using cosine annealing
        lr = cosine_annealing_lr(epoch, num_epochs, args.lr0, args.lrf)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        print(f'Epoch {epoch + 1}/{num_epochs} | Learning Rate: {lr:.6f}')

        # Train or validate phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set the model to training mode
            else:
                model.eval()  # Set the model to evaluation mode

            running_loss = 0.0
            corrects = 0

            # Loop through batches in the data loader
            for inputs, labels in tqdm(dataloaders[phase], desc=f'{phase.capitalize()} Epoch {epoch + 1}/{num_epochs}'):
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()  # Clear gradients

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)  # Forward pass
                    loss = criterion(outputs, labels)  # Calculate loss

                    if phase == 'train':
                        loss.backward()  # Backpropagation
                        optimizer.step()  # Update weights

                    _, preds = torch.max(outputs, 1)
                    running_loss += loss.item() * inputs.size(0)
                    corrects += torch.sum(preds == labels.data)

                    writer.add_scalar(f'{phase.capitalize()}/Loss', loss.item(), global_step)
                    global_step += 1

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = corrects.double() / len(dataloaders[phase].dataset)

            writer.add_scalar(f'{phase.capitalize()}/Accuracy', epoch_acc, epoch)

            print(f'Epoch {epoch + 1}/{num_epochs} | {phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Save the best model based on validation accuracy and loss
            if phase == 'val' and epoch_loss < best_loss and epoch_acc >= best_acc:
                best_acc = epoch_acc
                best_loss = epoch_loss
                save_path = os.path.join(model_dir, f'{experiment_name}_best_model.pth')
                torch.save({
                    'model_state_dict': model.state_dict(),
                    # Other relevant keys
                }, save_path)
                print(f'Saved best model with accuracy: {best_acc:.4f} and loss: {best_loss:.4f} to {save_path}')

    print('Training finished!')


# Main function
def main(args):
    """
    Main function to set up and run the training process.

    Args:
        args (argparse.Namespace): Parsed command line arguments.

    Returns:
        None
    """
    # Data transformations for training and validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomRotation(degrees=37),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.RandomVerticalFlip(),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
    }

    # Load datasets using ImageFolder
    image_datasets = {x: datasets.ImageFolder(os.path.join(args.data_dir, x),
                                              transform=data_transforms[x])
                      for x in ['train', 'val']}

    # Create data loaders for training and validation
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.batch_size,
                                                  shuffle=True, num_workers=4)
                   for x in ['train', 'val']}

    # Choose device for training (cuda if available, else cpu)
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")

    # Load a pretrained ResNet-50 model and modify the final fully connected layers
    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 256),
        nn.ReLU(),
        nn.Dropout(0.45),  # Adding dropout
        nn.Linear(256, args.num_classes)
    )
    model = model.to(device)

    # Define loss function and optimization algorithm
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr0, momentum=0.9, weight_decay=0.001)  # Adding weight decay

    # Set up logging directories
    base_log_dir = os.path.join(args.log_dir, "basic_classification")
    if not os.path.exists(base_log_dir):
        os.makedirs(base_log_dir)

    # Create a unique experiment name
    experiment_name = f"lr0_{args.lr0}_lrf_{args.lrf}_batch_{args.batch_size}"
    experiment_count = 1
    while True:
        log_dir = os.path.join(base_log_dir, f"{experiment_name}_exp_{experiment_count}")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            break
        experiment_count += 1

    # Create directory for saving trained models
    model_dir = os.path.join(args.model_dir, f"{experiment_name}_exp_{experiment_count}")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Create a TensorBoard writer for logging
    writer = SummaryWriter(log_dir=log_dir)

    # Start training the model
    train_model(model, dataloaders, criterion, optimizer, args.num_epochs, device, writer, model_dir, experiment_name)
    writer.close()


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train a ResNet-50 model for binary classification")
    parser.add_argument("--data_dir", type=str, default="path_to_data_directory",
                        help="Path to the directory containing train, val, and test folders")
    parser.add_argument("--num_classes", type=int, default=2, help="Number of classes (2 for binary)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr0", type=float, default=0.001, help="Initial learning rate")
    parser.add_argument("--lrf", type=float, default=0.00001, help="Final learning rate")
    parser.add_argument("--log_dir", type=str, default="logs", help="Base directory for TensorBoard logs")
    parser.add_argument("--model_dir", type=str, default="models", help="Directory to save model checkpoints")
    parser.add_argument("--gpu", action="store_true", help="Use GPU if available")
    args = parser.parse_args()

    # Create log directory if it doesn't exist
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    # Call the main function to start training
    main(args)
