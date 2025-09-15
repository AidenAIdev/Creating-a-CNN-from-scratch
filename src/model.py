import torch
import torch.nn as nn


# define the CNN architecture
class MyModel(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.7) -> None:

        super().__init__()

        # YOUR CODE HERE
        # Define a CNN architecture. Remember to use the variable num_classes
        # to size appropriately the output of your classifier, and if you use
        # the Dropout layer, use the variable "dropout" to indicate how much
        # to use (like nn.Dropout(p=dropout))
        
        # Convolutional layers with batch normalization
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Global Average Pooling for better performance
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Dropout layers
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout * 0.5)  # Less dropout for final layer
        
        # Fully connected layers with better architecture
        self.fc1 = nn.Linear(512, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_classes)
        
        # Activation function
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # YOUR CODE HERE: process the input tensor through the
        # feature extractor, the pooling and the final linear
        # layers (if appropriate for the architecture chosen)
        
        # First convolutional block
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        
        # Second convolutional block
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        
        # Third convolutional block
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        
        # Fourth convolutional block
        x = self.pool(self.relu(self.bn4(self.conv4(x))))
        
        # Global average pooling
        x = self.global_pool(x)
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)
        
        # First fully connected layer with dropout
        x = self.dropout1(self.relu(self.fc1(x)))
        
        # Second fully connected layer with dropout
        x = self.dropout2(self.relu(self.fc2(x)))
        
        # Final classification layer (no activation, will be handled by loss function)
        x = self.fc3(x)
        
        return x


######################################################################################
#                                     TESTS
######################################################################################
try:
    import pytest
except Exception:
    pytest = None


if pytest is not None:
    @pytest.fixture(scope="session")
    def data_loaders():
        from .data import get_data_loaders

        return get_data_loaders(batch_size=2)


    def test_model_construction(data_loaders):

        model = MyModel(num_classes=23, dropout=0.3)

        dataiter = iter(data_loaders["train"])
        images, labels = dataiter.next()

        out = model(images)

        assert isinstance(
            out, torch.Tensor
        ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

        assert out.shape == torch.Size(
            [images.shape[0], 23]
        ), f"Expected an output tensor of size (batch_size, n_classes), got {out.shape}"


    def test_model_construction_2(data_loaders):

        model = MyModel(num_classes=50, dropout=0.3)

        dataiter = iter(data_loaders["train"])
        images, labels = dataiter.next()

        out = model(images)

        assert isinstance(
            out, torch.Tensor
        ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

        assert out.shape == torch.Size(
            [images.shape[0], 50]
        ), f"Expected an output tensor of size (batch_size, n_classes), got {out.shape}"
