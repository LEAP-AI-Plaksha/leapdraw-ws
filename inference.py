import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf

def pad_image_to_square(img):
    # Get the dimensions of the image
    width, height = img.size
    max_dim = max(width, height)

    # Calculate padding
    padding_left = (max_dim - width) // 2
    padding_top = (max_dim - height) // 2
    padding_right = max_dim - width - padding_left
    padding_bottom = max_dim - height - padding_top

    # Apply padding
    padding = (padding_left, padding_top, padding_right, padding_bottom)
    img_with_padding = ImageOps.expand(img, padding, fill=(255, 255, 255))  # Fill with white color

    return img_with_padding

def preprocess_image(img):
    """Preprocess the image for prediction."""
    # Convert to grayscale
    img = img.convert('L')

    # Choose the resampling filter based on the Pillow version
    try:
        resample_filter = Image.Resampling.LANCZOS
    except AttributeError:
        resample_filter = Image.LANCZOS

    # Resize to 32x32
    img = img.resize((32, 32), resample=resample_filter)

    # Invert colors if necessary
    img = ImageOps.invert(img)

    # Convert to numpy array
    img_array = np.array(img)

    # Normalize pixel values
    img_normalized = img_array.astype('float32') / 255.0

    # Expand dimensions
    img_expanded = np.expand_dims(img_normalized, axis=(0, -1))

    return img_expanded

def predict_from_image(model_path, img, categories_file='categories.txt'):
    # Load categories
    with open(categories_file, 'r') as f:
        categories = [line.strip() for line in f.readlines()]

    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get model input details
    input_details = interpreter.get_input_details()
    input_dtype = input_details[0]['dtype']

    # Preprocess the image
    preprocessed_image = preprocess_image(img)
    preprocessed_image = preprocessed_image.astype(input_dtype)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], preprocessed_image)
    interpreter.invoke()

    # Get predictions
    output_details = interpreter.get_output_details()
    predictions = interpreter.get_tensor(output_details[0]['index'])

    predicted_index = np.argmax(predictions, axis=1)[0]
    confidence = predictions[0][predicted_index]
    predicted_category = categories[predicted_index]

    return predicted_category, confidence
