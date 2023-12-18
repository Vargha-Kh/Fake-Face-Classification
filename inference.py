from models import load_model
from torchvision import models, transforms
import torch
import torch.nn.functional as F
from PIL import Image


def preprocess_image(image_path):
    image = Image.open(image_path)
    preprocess = transforms.Compose([
        transforms.Resize((229, 299)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)
    return input_batch


def inference(image_path, device='cpu', model_path='./best_classification.pth'):
    # Ensure the model is in evaluation mode

    model = load_model("regnet")
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()

    input_batch = preprocess_image(image_path)
    # Move the input and model to the device
    input_batch = input_batch.to(device)

    # Perform inference
    with torch.no_grad():
        output = model(input_batch)

    # Convert the output to probabilities using softmax
    probabilities = F.softmax(output[0], dim=0)

    # Get the predicted class index
    predicted_class = torch.argmax(probabilities).item()

    # Display the results
    print(f"Predicted class: {predicted_class}")
    print(f"Class probabilities: {probabilities}")


if __name__ == "__main__":
    img_path = "sample_1.jpg"
    inference(img_path)
