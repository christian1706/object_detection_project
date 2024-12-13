import os 
import torch
from PIL import Image
import config
import matplotlib.pyplot as plt
import matplotlib.patches as patches
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
    transform = get_transform()  # Assuming `get_transform` handles both train and eval transforms
    transformed_image = transform(image)

    # Add batch dimension and move to the appropriate device
    transformed_image = transformed_image.squeeze(0).to(device)

    # Perform inference
    with torch.no_grad():
        outputs = model([transformed_image])

    return outputs[0], image

# Define the function to plot bounding boxes, scores, and labels
def plot_predictions(image, outputs, confidence_threshold=0.7):
    fig, ax = plt.subplots(1, figsize=(12, 12))
    ax.imshow(image)

    # Extract boxes, scores, and labels
    boxes = outputs['boxes'].cpu().numpy()
    scores = outputs['scores'].cpu().numpy()
    labels = outputs['labels'].cpu().numpy()  # Assuming 'labels' key exists in the outputs

    for box, score, label in zip(boxes, scores, labels):
        if score >= confidence_threshold:
            x_min, y_min, x_max, y_max = box

            # Draw the bounding box
            rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, 
                                      linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

            # Add the label and score text
            label_text = f"Label: {label}, Score: {score:.2f}"
            ax.text(x_min, y_min - 5, label_text, color='red', fontsize=12, 
                    bbox=dict(facecolor='white', alpha=0.5))

    plt.axis('off')
    plt.show()

# Define the main testing script
def main():
    # Configuration
    model_path = config.saved_model  # Replace with the path to your saved model
    image_path = config.test_image  # Replace with the path to your input image
    num_classes = config.num_classes  # Number of classes the model was trained on

    # Device configuration
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Load the model
    model = load_trained_model(model_path, num_classes, device)

    # Test the model on an input image
    outputs, original_image = test_model_on_image(model, image_path, device)

    # Display the results (customize this as needed)
    print("Model outputs:", outputs)

    # Plot the image with predictions
    plot_predictions(original_image, outputs, confidence_threshold=0.7)

if __name__ == "__main__":
    main()
