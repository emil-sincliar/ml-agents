import os
import numpy as np
from joblib import dump, load
import shutil

def split_model(model_path, chunk_size_mb=90, output_dir='model_chunks'):
    """
    Split a large model file into smaller chunks
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Read the model file in binary mode
    with open(model_path, 'rb') as f:
        data = f.read()
    
    # Calculate number of chunks needed
    chunk_size = chunk_size_mb * 1024 * 1024  # Convert MB to bytes
    total_chunks = (len(data) + chunk_size - 1) // chunk_size
    
    # Split the file into chunks
    for i in range(total_chunks):
        chunk = data[i * chunk_size : (i + 1) * chunk_size]
        chunk_path = os.path.join(output_dir, f'model_part_{i:03d}.bin')
        with open(chunk_path, 'wb') as f:
            f.write(chunk)
    
    # Save metadata about the original file
    metadata = {
        'original_size': len(data),
        'num_chunks': total_chunks,
        'chunk_size': chunk_size
    }
    dump(metadata, os.path.join(output_dir, 'metadata.joblib'))
    
    print(f'Model split into {total_chunks} chunks in {output_dir}/')
    return total_chunks

def combine_model(chunks_dir='model_chunks', output_path=None):
    """
    Combine model chunks back into a single file
    """
    # Load metadata
    metadata = load(os.path.join(chunks_dir, 'metadata.joblib'))
    total_chunks = metadata['num_chunks']
    
    # If output path is not specified, use the original filename
    if output_path is None:
        output_path = 'reconstructed_model.joblib'
    
    # Combine chunks
    with open(output_path, 'wb') as outfile:
        for i in range(total_chunks):
            chunk_path = os.path.join(chunks_dir, f'model_part_{i:03d}.bin')
            with open(chunk_path, 'rb') as chunk_file:
                shutil.copyfileobj(chunk_file, outfile)
    
    print(f'Model reconstructed to {output_path}')
    
    # Verify file size
    actual_size = os.path.getsize(output_path)
    if actual_size != metadata['original_size']:
        raise ValueError(f'Size mismatch! Expected {metadata["original_size"]} bytes but got {actual_size} bytes')
    
    return output_path

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python model_utils.py [split|combine] [input_path] [output_path]")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == 'split':
        if len(sys.argv) < 3:
            print("Please provide the model path to split")
            sys.exit(1)
        model_path = sys.argv[2]
        output_dir = sys.argv[3] if len(sys.argv) > 3 else 'model_chunks'
        split_model(model_path, output_dir=output_dir)
    
    elif command == 'combine':
        chunks_dir = sys.argv[2] if len(sys.argv) > 2 else 'model_chunks'
        output_path = sys.argv[3] if len(sys.argv) > 3 else None
        combine_model(chunks_dir, output_path)
    
    else:
        print("Unknown command. Use 'split' or 'combine'")
        sys.exit(1)
