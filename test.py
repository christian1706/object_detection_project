import os
import torch
from PIL import Image
import config
import matplotlib.pyplot as plt
from utils import get_model, get_transform, inference

# Define the function to load the trained model
def load_trained_model(model_path, num_classes, device):
    model = get_model(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device)['model_state_dict'])
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    return model

# Define the function to test the model on an input image
def test_model_on_image(model, image_path, device):
    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    transform = get_transform(train=False)  # Assuming `get_transform` handles both train and eval transforms
    transformed_image = transform(image)

    # Add batch dimension and move to the appropriate device
    transformed_image = transformed_image.unsqueeze(0).to(device)

    # Perform inference
    with torch.no_grad():
        outputs = model([transformed_image])

    return outputs[0], image

# Define the main testing script
def main():
    # Configuration
    model_path =config.saved_model # Replace with the path to your saved model
    image_path = "path/to/your/image.jpg"  # Replace with the path to your input image
    num_classes = config.num_classes        # Number of classes the model was trained on

    # Device configuration
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Load the model
    model = load_trained_model(model_path, num_classes, device)

    # Test the model on an input image
    outputs, original_image = test_model_on_image(model, image_path, device)

    # Display the results (customize this as needed)
    print("Model outputs:", outputs)

    # Optionally visualize the image with predictions
    plt.figure(figsize=(8, 8))
    plt.imshow(original_image)
    plt.title("Model Predictions")
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    main()
