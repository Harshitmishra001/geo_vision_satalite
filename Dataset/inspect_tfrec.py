import tensorflow as tf
import sys

def inspect_tfrecord(file_path):
    print(f"Inspecting {file_path}...\n")
    
    # Load the TFRecord dataset
    raw_dataset = tf.data.TFRecordDataset(file_path)
    
    # Take just the very first record to inspect
    for raw_record in raw_dataset.take(1):
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        
        # Print the keys (features) hidden inside
        print("--- TFRecord Keys Found ---")
        for key, feature in example.features.feature.items():
            # Figure out what type of data it is (Bytes, Float, or Int)
            kind = feature.WhichOneof('kind')
            print(f"Key: '{key}' | Data Type: {kind}")

if __name__ == "__main__":
    # Make sure this path points to your actual train.tfrec file
    inspect_tfrecord("train.tfrec")