import tensorflow as tf
import cv2
import numpy as np
import os

def decode_serialized_tensor(byte_data):
    """Attempts to decode a serialized TensorFlow tensor."""
    # First, try standard 8-bit integers (0-255)
    try:
        tensor = tf.io.parse_tensor(byte_data, out_type=tf.uint8)
        return tensor.numpy()
    except:
        pass
        
    # Next, try float32 (Deep learning models often normalize images between 0.0 and 1.0)
    try:
        tensor = tf.io.parse_tensor(byte_data, out_type=tf.float32)
        arr = tensor.numpy()
        # Scale it back to standard image pixels
        if arr.max() <= 1.0:
            arr = (arr * 255.0)
        return arr.astype(np.uint8)
    except:
        return None

def extract_tfrecord(tfrec_path, output_dir="dataset", num_samples=10):
    print(f"Creating directories in {output_dir}...")
    os.makedirs(f"{output_dir}/baseline", exist_ok=True)
    os.makedirs(f"{output_dir}/current", exist_ok=True)
    os.makedirs(f"{output_dir}/ground_truth", exist_ok=True)

    feature_description = {
        'img1': tf.io.FixedLenFeature([], tf.string),
        'img2': tf.io.FixedLenFeature([], tf.string),
        'ref':  tf.io.FixedLenFeature([], tf.string),
    }

    def _parse_function(example_proto):
        return tf.io.parse_single_example(example_proto, feature_description)

    print(f"Decoding Serialized Tensors from {tfrec_path}...")
    raw_dataset = tf.data.TFRecordDataset(tfrec_path)
    parsed_dataset = raw_dataset.map(_parse_function)

    extracted_count = 0
    for i, parsed_record in enumerate(parsed_dataset):
        if extracted_count >= num_samples:
            break
            
        # Use our new tensor parser
        img1 = decode_serialized_tensor(parsed_record['img1'])
        img2 = decode_serialized_tensor(parsed_record['img2'])
        ref = decode_serialized_tensor(parsed_record['ref'])

        if img1 is None or img2 is None or ref is None:
            print(f"Skipping pair {i}: Could not parse as a valid Tensor.")
            continue

        # Convert RGB to BGR for OpenCV
        if len(img1.shape) == 3 and img1.shape[2] == 3:
            img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
            img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)

        # Save to folders
        cv2.imwrite(f"{output_dir}/baseline/pair_{i}.png", img1)
        cv2.imwrite(f"{output_dir}/current/pair_{i}.png", img2)
        cv2.imwrite(f"{output_dir}/ground_truth/pair_{i}.png", ref)
        
        extracted_count += 1
        print(f"Extracted pair {i} successfully!")

    print(f"\n[SUCCESS] Extracted {extracted_count} image pairs to the '{output_dir}' folder!")

if __name__ == "__main__":
    extract_tfrecord("train.tfrec")