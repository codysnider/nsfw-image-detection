import argparse
import torch
from PIL import Image, UnidentifiedImageError
from transformers import AutoModelForImageClassification, ViTImageProcessor

# Use GPU if available, otherwise fallback to CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device}")

def load_model():
    """Loads the FalconsAI NSFW Image Detection model."""
    print("Loading model...")
    model = AutoModelForImageClassification.from_pretrained("Falconsai/nsfw_image_detection")
    processor = ViTImageProcessor.from_pretrained("Falconsai/nsfw_image_detection")

    model.to(device).eval()  # Move to device and set eval mode
    return model, processor

@torch.no_grad()
def classify_image(model, processor, img_path):
    """Classifies an image as NSFW or SFW."""
    try:
        img = Image.open(img_path)
    except (FileNotFoundError, UnidentifiedImageError) as e:
        print(f"Error: Unable to open image '{img_path}': {e}")
        return None

    inputs = processor(images=img, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}  # Move tensors to device

    logits = model(**inputs).logits
    predicted_label = logits.argmax(-1).item()

    label = model.config.id2label[predicted_label]
    print(f"Predicted Label: {label}")

    return label

def main():
    parser = argparse.ArgumentParser(description="NSFW Image Classification")
    parser.add_argument("--image", type=str, required=True, help="Path to the image file")
    args = parser.parse_args()

    model, processor = load_model()
    classify_image(model, processor, args.image)

if __name__ == "__main__":
    main()
