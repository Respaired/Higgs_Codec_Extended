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
import librosa
import torch
import torch.nn.functional as F
import numpy as np
import json
import torch
from higgs_audio_tokenizer import HiggsAudioTokenizer
import torch
import torch.nn as nn
import warnings

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
    e_semantic_input = model.get_regress_target(x_batch).detach()
    e_semantic = model.encoder_semantic(e_semantic_input.transpose(1, 2))
    e_acoustic = model.encoder(x_batch)
    
    if e_acoustic.shape[2] != e_semantic.shape[2]:
        pad_size = 160 * model.semantic_downsample_factor
        
        x_slice = x_batch[:, 0, :]
        
        x_padded = F.pad(x_slice, (pad_size, pad_size))
        
        e_acoustic = model.encoder(x_padded.unsqueeze(1))
    
    min_len = min(e_acoustic.shape[2], e_semantic.shape[2])
    e_acoustic = e_acoustic[:, :, :min_len]
    e_semantic = e_semantic[:, :, :min_len]
    
    e = torch.cat([e_acoustic, e_semantic], dim=1)
    e = model.fc_prior(e.transpose(1, 2))
    
    if model.quantizer_type == "RVQ":
        e = e.transpose(1, 2)
        _, codes, _, _ = model.quantizer(e, model.frame_rate, None)
        codes = codes.permute(1, 0, 2)
    else:
        quantized, codes = model.quantizer(e)
        codes = codes.permute(0, 2, 1)
    
    return EncodedResult(audio_codes=codes)


def fix_all_inference_issues(model):
    device = next(model.parameters()).device
    
    model.eval()
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, nn.Module):
                module.eval()
                if hasattr(module, 'training'):
                    module.training = False
    
    if hasattr(model, 'semantic_model'):
        print("Fixing semantic model...")
        
        model.semantic_model = model.semantic_model.to(device)
        model.semantic_model.eval()
        
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
        
        if hasattr(model.semantic_model, 'encoder'):
            model.semantic_model.encoder.gradient_checkpointing = False
            if hasattr(model.semantic_model.encoder, 'layers'):
                for layer in model.semantic_model.encoder.layers:
                    if hasattr(layer, 'gradient_checkpointing'):
                        layer.gradient_checkpointing = False
    
    def set_dropout_eval(module):
        if isinstance(module, nn.Dropout):
            module.eval()
            module.training = False
        for child in module.children():
            set_dropout_eval(child)
    
    set_dropout_eval(model)
    
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return model


def inference_pipeline(checkpoint_path, config_path, device='cuda'):
    print("Loading config...")
    with open(config_path, 'r') as f:
        config = json.load(f)
    
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
    
    print("Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict, strict=False)
    
    print("Fixing inference issues...")
    model = fix_all_inference_issues(model)
    
    return model


warnings.filterwarnings("ignore")

OUTPUT_DIR = "/home/ubuntu/data_boson_44.1khz"
BATCH_SIZE = 32
SAMPLE_RATE = 44100
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATASET_PATH = "/home/ubuntu/ttsar/Layla/src_bpe_2/Qanary_data"

print(f"Using device: {DEVICE}")

os.chdir("/home/ubuntu/ttsar/boson_audio_codec/audio_processing")

from datasets import load_from_disk

print(f"Loading dataset from: {DATASET_PATH}")
ds = load_from_disk(DATASET_PATH)
print(f"Dataset info: {ds}")

columns_to_remove = ['spk', 'duration', 'codes', 'input_ids', 'attention_mask']
existing_columns = [col for col in columns_to_remove if col in ds.column_names]
if existing_columns:
    ds = ds.remove_columns(existing_columns)

df = ds.to_pandas()
print(f"Loaded {len(df)} files from dataset")

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Output directory '{OUTPUT_DIR}' is ready.")

print("Checking for already processed files...")


def get_output_path(audio_path):
    base_name = Path(audio_path).stem
    return os.path.join(OUTPUT_DIR, f"{base_name}.pt")


original_count = len(df)
df['output_exists'] = df['filename'].apply(lambda x: os.path.exists(get_output_path(x)))
df_filtered = df[~df['output_exists']].copy()
skipped_count = original_count - len(df_filtered)

print(f"Found {skipped_count} already processed files. Skipping them.")
print(f"Processing {len(df_filtered)} remaining files.")

if len(df_filtered) == 0:
    print("All files have already been processed!")
    exit()

print("Loading Higgs Audio Tokenizer model...")
from transformers import HubertModel
from higgs_audio_tokenizer import HiggsAudioTokenizer

checkpoint_path = '/home/ubuntu/ttsar/boson_audio_codec/audio_processing/outputs_CQT/checkpoints/step_99000.pth'
config_path = '/home/ubuntu/ttsar/boson_audio_codec/audio_processing/config copy.json'
device = 'cuda'

model = inference_pipeline(checkpoint_path, config_path, device)
_ = model.eval()
model = remove_weight_norms_from_model(model)
print(f"Model loaded on {DEVICE}")

hop_length = model.hop_length
print(f"Encoder hop length: {hop_length}")

print(f"\nStarting batch processing with batch size {BATCH_SIZE}...")

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
        
        for filename in batch_filenames:
            output_path = get_output_path(filename)
            
            if os.path.exists(output_path):
                continue
            
            try:
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
        
        max_len = max(len(x) for x in batch_audio)
        padded_batch = []
        
        for audio in batch_audio:
            pad_len = max_len - len(audio)
            if pad_len > 0:
                audio = F.pad(audio, (0, pad_len), mode='constant', value=0)
            padded_batch.append(audio)
        
        batch_tensor = torch.stack(padded_batch, dim=0)
        batch_tensor = batch_tensor.unsqueeze(1)
        batch_tensor = batch_tensor.to(DEVICE)
        
        try:
            encoded = encode_batch(model, batch_tensor)
            codes = encoded.audio_codes
            
            for idx, (output_path, orig_len) in enumerate(zip(batch_outputs, batch_lengths)):
                true_code_len = int(np.ceil(orig_len / hop_length))
                
                item_codes = codes[idx, :, :true_code_len].cpu()
                
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

final_count = len(list(Path(OUTPUT_DIR).glob("*.pt")))
print(f"Total .pt files in output: {final_count}")