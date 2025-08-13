# #!/usr/bin/env python3
# """
# Audio Processing Script for Boson Codes
# Processes audio files in parallel using Higgs Audio Tokenizer
# and saves encoded representations as .pt files.
# """

# import os
# import sys
# import json
# import torch
# import librosa
# import numpy as np
# import warnings
# import argparse
# from pathlib import Path
# from multiprocessing import Pool
# from tqdm import tqdm

# from datasets import load_from_disk
# from higgs_audio_tokenizer import HiggsAudioTokenizer

# # Suppress PyTorch FutureWarnings
# warnings.filterwarnings("ignore", category=FutureWarning)

# # Global configuration
# DEFAULT_OUTPUT_DIR = "/home/ubuntu/boson_codes"
# DEFAULT_NUM_CORES = 48
# DEFAULT_SAMPLE_RATE = 44100
# DEFAULT_DATASET_PATH = "/home/ubuntu/ttsar/Layla/src_bpe_2/data"

# # Model paths
# CONFIG_PATH = "/home/ubuntu/.cache/huggingface/hub/models--bosonai--higgs-audio-v2-tokenizer/snapshots/9d4988fbd4ad07b4cac3a5fa462741a41810dbec/config.json"
# MODEL_PATH = "/home/ubuntu/.cache/huggingface/hub/models--bosonai--higgs-audio-v2-tokenizer/snapshots/9d4988fbd4ad07b4cac3a5fa462741a41810dbec/model.pth"

# # Global model variable (initialized in each worker)
# model = None


# def init_worker():
#     """Initialize model once per worker process."""
#     global model
#     device = 'cpu'
    
#     # Load config
#     with open(CONFIG_PATH, 'r') as f:
#         config = json.load(f)
    
#     # Initialize model
#     model = HiggsAudioTokenizer(
#         **config,
#         device=device,
#     )
    
#     # Load weights
#     parameter_dict = torch.load(MODEL_PATH, map_location=device)
#     _ = model.load_state_dict(parameter_dict, strict=False)
#     model = model.to(device)
#     _ = model.eval()
    
#     print(f"Model loaded in worker {os.getpid()}")


# def process_audio_file(args):
#     """Process a single audio file using pre-loaded model."""
#     filename, output_dir, sample_rate = args
    
#     try:
#         # Output filename - same name, just change extension to .pt
#         base_name = Path(filename).stem
#         output_path = os.path.join(output_dir, f"{base_name}.pt")
        
#         # Skip if exists (double-check in case of race conditions)
#         if os.path.exists(output_path):
#             return ("skipped", filename)
        
#         # Load and process audio
#         wav, sr = librosa.load(filename, sr=sample_rate)
#         wav = torch.from_numpy(wav).unsqueeze(0).float().to('cpu')
        
#         # Encode using the pre-loaded model
#         with torch.no_grad():
#             encoded = model._xcodec_encode(wav.unsqueeze(0))
        
#         # Save codes only
#         torch.save(encoded.audio_codes, output_path)
        
#         return ("success", filename)
        
#     except Exception as e:
#         return ("error", filename, str(e))


# def load_dataset(dataset_path):
#     """Load and prepare the dataset."""
#     print(f"Loading dataset from: {dataset_path}")
#     ds = load_from_disk(dataset_path)
#     print(f"Dataset info: {ds}")
    
#     # Remove unnecessary columns
#     columns_to_remove = ['spk', 'duration', 'codes', 'input_ids', 'attention_mask']
#     existing_columns = [col for col in columns_to_remove if col in ds.column_names]
#     if existing_columns:
#         ds = ds.remove_columns(existing_columns)
#         print(f"Removed columns: {existing_columns}")
    
#     # Convert to pandas DataFrame
#     df = ds.to_pandas()
#     print(f"Loaded {len(df)} files from dataset")
#     return df


# def main(args):
#     """Main processing function."""
#     # Change to audio processing directory
#     os.chdir("/home/ubuntu/ttsar/boson_audio_codec/audio_processing")
#     print(f"Working directory: {os.getcwd()}")
    
#     # Create output directory
#     os.makedirs(args.output_dir, exist_ok=True)
#     print(f"Output directory: {args.output_dir}")
    
#     # Check if model files exist
#     if not os.path.exists(CONFIG_PATH):
#         print(f"Error: Config file not found at {CONFIG_PATH}")
#         sys.exit(1)
#     if not os.path.exists(MODEL_PATH):
#         print(f"Error: Model file not found at {MODEL_PATH}")
#         sys.exit(1)
    
#     # Load dataset
#     df = load_dataset(args.dataset_path)
    
#     # Get filenames from dataframe
#     all_filenames = df['filename'].tolist()
    
#     # Pre-filter to exclude already processed files
#     filenames_to_process = []
#     already_processed = []
    
#     print(f"\nChecking for already processed files...")
#     for filename in all_filenames:
#         base_name = Path(filename).stem
#         output_path = os.path.join(args.output_dir, f"{base_name}.pt")
#         if os.path.exists(output_path):
#             already_processed.append(filename)
#         else:
#             filenames_to_process.append(filename)
    
#     print(f"\nTotal files: {len(all_filenames)}")
#     print(f"Already processed: {len(already_processed)}")
#     print(f"To process: {len(filenames_to_process)}")
    
#     if len(filenames_to_process) == 0:
#         print("\nAll files have already been processed!")
#         return
    
#     print(f"\nProcessing {len(filenames_to_process)} files using {args.num_cores} cores...")
#     print(f"Sample rate: {args.sample_rate} Hz")
    
#     # Prepare arguments for multiprocessing
#     process_args = [(filename, args.output_dir, args.sample_rate) 
#                     for filename in filenames_to_process]
    
#     # Process in parallel with model reuse
#     with Pool(processes=args.num_cores, initializer=init_worker) as pool:
#         results = list(tqdm(
#             pool.imap(process_audio_file, process_args, chunksize=args.chunksize),
#             total=len(filenames_to_process),
#             desc="Processing audio files"
#         ))
    
#     # Count results
#     processed = sum(1 for r in results if r[0] == "success")
#     skipped = sum(1 for r in results if r[0] == "skipped")
#     errors = sum(1 for r in results if r[0] == "error")
    
#     print(f"\nProcessing complete!")
#     print(f"  Successfully processed: {processed}")
#     print(f"  Previously processed: {len(already_processed)}")
#     print(f"  Skipped (race condition): {skipped}")
#     print(f"  Errors: {errors}")
    
#     # Show errors if any
#     if errors > 0:
#         print("\nErrors encountered:")
#         error_log_path = os.path.join(args.output_dir, "processing_errors.log")
#         with open(error_log_path, 'w') as f:
#             for r in results:
#                 if r[0] == "error":
#                     error_msg = f"{r[1]}: {r[2]}"
#                     print(f"  {error_msg}")
#                     f.write(error_msg + "\n")
#         print(f"\nError log saved to: {error_log_path}")
    
#     # Show summary of all processed files
#     total_processed_files = len(list(Path(args.output_dir).glob("*.pt")))
#     print(f"\nTotal .pt files in {args.output_dir}: {total_processed_files}")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(
#         description="Process audio files using Higgs Audio Tokenizer and save as .pt files"
#     )
    
#     parser.add_argument(
#         "--dataset-path", 
#         type=str, 
#         default=DEFAULT_DATASET_PATH,
#         help=f"Path to the dataset (default: {DEFAULT_DATASET_PATH})"
#     )
    
#     parser.add_argument(
#         "--output-dir", 
#         type=str, 
#         default=DEFAULT_OUTPUT_DIR,
#         help=f"Output directory for .pt files (default: {DEFAULT_OUTPUT_DIR})"
#     )
    
#     parser.add_argument(
#         "--num-cores", 
#         type=int, 
#         default=DEFAULT_NUM_CORES,
#         help=f"Number of CPU cores to use (default: {DEFAULT_NUM_CORES})"
#     )
    
#     parser.add_argument(
#         "--sample-rate", 
#         type=int, 
#         default=DEFAULT_SAMPLE_RATE,
#         help=f"Sample rate for audio processing (default: {DEFAULT_SAMPLE_RATE})"
#     )
    
#     parser.add_argument(
#         "--chunksize", 
#         type=int, 
#         default=1,
#         help="Chunksize for multiprocessing pool (default: 1)"
#     )
    
#     args = parser.parse_args()
    
#     # Run main processing
#     try:
#         main(args)
#     except KeyboardInterrupt:
#         print("\n\nProcessing interrupted by user")
#         sys.exit(1)
#     except Exception as e:
#         print(f"\n\nError: {e}")
#         sys.exit(1)

#!/usr/bin/env python3
"""
GPU Batch Processing Script for Boson Codes with Dataset Loading
"""

import os
import sys
import json
import torch
import torch.nn.functional as F
import librosa
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings
from torch.nn.utils import remove_weight_norm, weight_norm


# from boson_multimodal.audio_processing.higgs_audio_tokenizer import load_higgs_audio_tokenizer
# model = load_higgs_audio_tokenizer("bosonai/higgs-audio-v2-tokenizer")
import librosa
import torch
import torch.nn.functional as F
import numpy as np
import json
import torch

from higgs_audio_tokenizer import HiggsAudioTokenizer
# model = load_higgs_audio_tokenizer("bosonai/higgs-audio-v2-tokenizer")

import torch
import torch.nn as nn
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

def remove_weight_norms_from_model(model):
    for module in model.modules():
        try:
            remove_weight_norm(module)
        except:
            continue
    return model


class EncodedResult:
    def __init__(self, audio_codes):
        self.audio_codes = audio_codes

def encode_batch(model, x_batch):
    """
    Encodes a batch of audio tensors using the HiggsAudioTokenizer model.
    Args:
        model: The loaded HiggsAudioTokenizer model.
        x_batch: A tensor of shape [B, 1, T]
    """
    # Acoustic and Semantic Feature Extraction
    e_semantic_input = model.get_regress_target(x_batch).detach()
    e_semantic = model.encoder_semantic(e_semantic_input.transpose(1, 2))
    e_acoustic = model.encoder(x_batch)

    # This block contains the fix for batch processing
    if e_acoustic.shape[2] != e_semantic.shape[2]:
        pad_size = 160 * model.semantic_downsample_factor
        
        # 1. Remove channel dim, preserving batch dim -> [B, T]
        x_slice = x_batch[:, 0, :]
        
        # 2. Pad the tensor
        x_padded = F.pad(x_slice, (pad_size, pad_size))
        
        # 3. Re-add channel dim before passing to encoder -> [B, 1, T_padded]
        e_acoustic = model.encoder(x_padded.unsqueeze(1))

    # Ensure dimensions match before concatenating
    min_len = min(e_acoustic.shape[2], e_semantic.shape[2])
    e_acoustic = e_acoustic[:, :, :min_len]
    e_semantic = e_semantic[:, :, :min_len]

    # Remainder of the original encoding logic
    e = torch.cat([e_acoustic, e_semantic], dim=1)
    e = model.fc_prior(e.transpose(1, 2))

    if model.quantizer_type == "RVQ":
        e = e.transpose(1, 2)
        _, codes, _, _ = model.quantizer(e, model.frame_rate, None)
        codes = codes.permute(1, 0, 2)
    else: # RFSQ
        quantized, codes = model.quantizer(e)
        codes = codes.permute(0, 2, 1)

    return EncodedResult(audio_codes=codes)


def fix_all_inference_issues(model):
    """
    Comprehensive fix for all potential inference issues
    """
    device = next(model.parameters()).device
    
    # 1. Force everything to eval mode
    model.eval()
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, nn.Module):
                module.eval()
                if hasattr(module, 'training'):
                    module.training = False
    
    # 2. Fix semantic model specifically
    if hasattr(model, 'semantic_model'):
        print("Fixing semantic model...")
        
        # Move to correct device
        model.semantic_model = model.semantic_model.to(device)
        model.semantic_model.eval()
        
        # Disable ALL gradient checkpointing
        def disable_gradient_checkpointing(module):
            if hasattr(module, 'gradient_checkpointing'):
                module.gradient_checkpointing = False
            if hasattr(module, 'gradient_checkpointing_disable'):
                try:
                    module.gradient_checkpointing_disable()
                except:
                    pass
            for child in module.children():
                disable_gradient_checkpointing(child)
        
        disable_gradient_checkpointing(model.semantic_model)
        
        # For HuBERT specifically
        if hasattr(model.semantic_model, 'encoder'):
            model.semantic_model.encoder.gradient_checkpointing = False
            if hasattr(model.semantic_model.encoder, 'layers'):
                for layer in model.semantic_model.encoder.layers:
                    if hasattr(layer, 'gradient_checkpointing'):
                        layer.gradient_checkpointing = False
    
    # 3. Set all dropout to eval mode
    def set_dropout_eval(module):
        if isinstance(module, nn.Dropout):
            module.eval()
            module.training = False
        for child in module.children():
            set_dropout_eval(child)
    
    set_dropout_eval(model)
    
    # 4. Clear any cached computations
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return model

def inference_pipeline(checkpoint_path, config_path, device='cuda'):
    """
    Complete pipeline for inference with your trained model
    """
    # Load config
    print("Loading config...")
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Create model
    print("Creating model...")
    model = HiggsAudioTokenizer(
        n_filters=config['n_filters'],
        D=config['D'],
        target_bandwidths=config['target_bandwidths'],
        ratios=config['ratios'],
        sample_rate=config['sample_rate'],
        bins=config['bins'],
        n_q=config['n_q'],
        codebook_dim=config.get('codebook_dim', None),
        semantic_techer=config['semantic_techer'],
        device=device
    ).to(device)
    
    # Load checkpoint
    print("Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Remove 'module.' prefix if present (from DDP)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict, strict=False)
    
    # Fix all inference issues
    print("Fixing inference issues...")
    model = fix_all_inference_issues(model)

    
    return model



# # Add paths
# sys.path.insert(0, "/home/ubuntu/AP-BWE")

# Suppress warnings
warnings.filterwarnings("ignore")

# Configuration
OUTPUT_DIR = "/home/ubuntu/data_boson_44.1khz"
BATCH_SIZE = 32
SAMPLE_RATE = 44100
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATASET_PATH = "/home/ubuntu/ttsar/Layla/src_bpe_2/Qanary_data"

# # Model paths
# CONFIG_PATH = "/home/ubuntu/.cache/huggingface/hub/models--bosonai--higgs-audio-v2-tokenizer/snapshots/9d4988fbd4ad07b4cac3a5fa462741a41810dbec/config.json"
# MODEL_PATH = "/home/ubuntu/.cache/huggingface/hub/models--bosonai--higgs-audio-v2-tokenizer/snapshots/9d4988fbd4ad07b4cac3a5fa462741a41810dbec/model.pth"

# --- Setup ---
print(f"Using device: {DEVICE}")

# Change to working directory
os.chdir("/home/ubuntu/ttsar/boson_audio_codec/audio_processing")

# Load dataset
from datasets import load_from_disk


print(f"Loading dataset from: {DATASET_PATH}")
ds = load_from_disk(DATASET_PATH)
print(f"Dataset info: {ds}")

# Remove unnecessary columns
columns_to_remove = ['spk', 'duration', 'codes', 'input_ids', 'attention_mask']
existing_columns = [col for col in columns_to_remove if col in ds.column_names]
if existing_columns:
    ds = ds.remove_columns(existing_columns)

df = ds.to_pandas()
print(f"Loaded {len(df)} files from dataset")

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Output directory '{OUTPUT_DIR}' is ready.")

# --- Filter already processed ---
print("Checking for already processed files...")

def get_output_path(audio_path):
    base_name = Path(audio_path).stem
    return os.path.join(OUTPUT_DIR, f"{base_name}.pt")

# Filter
original_count = len(df)
df['output_exists'] = df['filename'].apply(lambda x: os.path.exists(get_output_path(x)))
df_filtered = df[~df['output_exists']].copy()
skipped_count = original_count - len(df_filtered)

print(f"Found {skipped_count} already processed files. Skipping them.")
print(f"Processing {len(df_filtered)} remaining files.")

if len(df_filtered) == 0:
    print("All files have already been processed!")
    exit()

# --- Load Model ---
print("Loading Higgs Audio Tokenizer model...")

from transformers import HubertModel
from higgs_audio_tokenizer import HiggsAudioTokenizer

# Load config
# with open(CONFIG_PATH, 'r') as f:
#     config = json.load(f)

# # Initialize model
# model = HiggsAudioTokenizer(
#     **config,
#     device=DEVICE,
# )

# Load weights
# parameter_dict = torch.load(MODEL_PATH, map_location=DEVICE)
# _ = model.load_state_dict(parameter_dict, strict=False)
# model = model.to(DEVICE)
# _ = model.eval()


checkpoint_path = '/home/ubuntu/ttsar/boson_audio_codec/audio_processing/outputs_CQT/checkpoints/step_99000.pth'
config_path = '/home/ubuntu/ttsar/boson_audio_codec/audio_processing/config copy.json'
device = 'cuda'
model = inference_pipeline(checkpoint_path, config_path, device)
_ = model.eval()

model = remove_weight_norms_from_model(model)

print(f"Model loaded on {DEVICE}")

# Get hop length
hop_length = model.hop_length
print(f"Encoder hop length: {hop_length}")

# --- Batch Processing ---
print(f"\nStarting batch processing with batch size {BATCH_SIZE}...")

# Process in batches
filenames = df_filtered['filename'].tolist()
total_processed = 0
total_errors = 0

with torch.no_grad():
    for batch_start in tqdm(range(0, len(filenames), BATCH_SIZE), desc="Processing batches"):
        batch_end = min(batch_start + BATCH_SIZE, len(filenames))
        batch_filenames = filenames[batch_start:batch_end]
        
        batch_audio = []
        batch_lengths = []
        batch_outputs = []
        
        # Load batch
        for filename in batch_filenames:
            output_path = get_output_path(filename)
            
            # Skip if exists (race condition check)
            if os.path.exists(output_path):
                continue
            
            try:
                # Load audio
                wav, _ = librosa.load(filename, sr=SAMPLE_RATE)
                wav_tensor = torch.from_numpy(wav).float()
                
                batch_audio.append(wav_tensor)
                batch_lengths.append(len(wav))
                batch_outputs.append(output_path)
                
            except Exception as e:
                print(f"\nError loading {filename}: {e}")
                total_errors += 1
                continue
        
        if not batch_audio:
            continue
        
        # Pad batch to same length
        max_len = max(len(x) for x in batch_audio)
        padded_batch = []
        
        for audio in batch_audio:
            pad_len = max_len - len(audio)
            if pad_len > 0:
                audio = F.pad(audio, (0, pad_len), mode='constant', value=0)
            # Don't add extra dimensions here, just collect the padded audio
            padded_batch.append(audio)
        
        # Convert list to tensor and add channel dimension
        # Stack along batch dimension to get [B, T]
        batch_tensor = torch.stack(padded_batch, dim=0)  # [B, T]
        # Add channel dimension
        batch_tensor = batch_tensor.unsqueeze(1)  # [B, 1, T]
        batch_tensor = batch_tensor.to(DEVICE)
        
        # Encode batch
        try:
            encoded = encode_batch(model, batch_tensor)
            codes = encoded.audio_codes  # [B, n_codebooks, T_compressed]
            
            # Save each item
            for idx, (output_path, orig_len) in enumerate(zip(batch_outputs, batch_lengths)):
                # Calculate true code length
                true_code_len = int(np.ceil(orig_len / hop_length))
                
                # Extract non-padded codes
                item_codes = codes[idx, :, :true_code_len].cpu()
                
                # Save
                torch.save(item_codes, output_path)
                total_processed += 1
                
        except Exception as e:
            print(f"\nError encoding batch: {e}")
            total_errors += len(batch_outputs)

print("\n" + "="*50)
print("PROCESSING COMPLETE!")
print("="*50)
print(f"Successfully processed: {total_processed} files")
print(f"Previously processed: {skipped_count} files")
print(f"Errors encountered: {total_errors} files")
print(f"Output directory: {OUTPUT_DIR}")

# Final count
final_count = len(list(Path(OUTPUT_DIR).glob("*.pt")))
print(f"Total .pt files in output: {final_count}")