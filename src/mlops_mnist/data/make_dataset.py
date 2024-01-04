import os

import torch




def create_dataset(image_files, target_files, data_path):
    """
    Function to create a dataset from the given list of file names
    """
    image_tensors = torch.cat([torch.load(os.path.join(data_path, file_name)) for file_name in image_files])
    target_tensors = torch.cat([torch.load(os.path.join(data_path, file_name)) for file_name in target_files])

    return image_tensors, target_tensors


def mnist():
    """Return train and test dataloaders for MNIST."""
    data_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "raw", "corruptmnist")

    # Create file name lists
    train_image_files = [f"train_images_{i}.pt" for i in range(6)]
    train_target_files = [f"train_target_{i}.pt" for i in range(6)]
    test_image_files = ["test_images.pt"]
    test_target_files = ["test_target.pt"]

    # Create datasets
    train_image_tensors, train_target_tensors = create_dataset(train_image_files, train_target_files, data_path)
    test_image_tensors, test_target_tensors = create_dataset(test_image_files, test_target_files, data_path)

    return train_image_tensors, train_target_tensors, test_image_tensors, test_target_tensors


def process(data):
    """
    Function to process the data by normalizing it
    """
    # Normalize the data
    mean = data.mean()
    std = data.std()
    data = (data - mean) / std

    return data


if __name__ == "__main__":
    # Get the data and process it
    train_image_tensors, train_target_tensors, test_image_tensors, test_target_tensors = mnist()

    # Save raw data as tensor dataset

    base_path = os.path.join("data", "raw")
    torch.save(train_image_tensors, os.path.join(base_path, "train_images.pt"))
    torch.save(train_target_tensors, os.path.join(base_path, "train_targets.pt"))
    torch.save(test_image_tensors, os.path.join(base_path, "test_images.pt"))
    torch.save(test_target_tensors, os.path.join(base_path, "test_targets.pt"))

    # Process the data
    train_image_tensors, test_image_tensors = process(train_image_tensors), process(test_image_tensors)

    # Save processed data
    base_path = os.path.join("data", "processed")
    torch.save(train_image_tensors, os.path.join(base_path, "train_images.pt"))
    torch.save(train_target_tensors, os.path.join(base_path, "train_targets.pt"))
    torch.save(test_image_tensors, os.path.join(base_path, "test_images.pt"))
    torch.save(test_target_tensors, os.path.join(base_path, "test_targets.pt"))
