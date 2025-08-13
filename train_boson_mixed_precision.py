import os
import json
import argparse
import random
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
import torchaudio
import librosa
from tqdm import tqdm
from audiotools import AudioSignal, STFTParams

# Import from the provided codebase
from higgs_audio_tokenizer import HiggsAudioTokenizer
from quantization.distrib import broadcast_tensors, sync_buffer, is_distributed, world_size, rank
from quantization.ddp_utils import set_random_seed, is_logging_process, get_timestamp

# Import DAC losses and discriminator
import sys
sys.path.append('.')  # Add current directory to path
from loss import L1Loss, MultiScaleSTFTLoss, MelSpectrogramLoss, GANLoss
from discriminator import Discriminator


class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Cosine scheduler with linear warmup"""
    def __init__(self, optimizer, warmup_steps, total_steps, eta_min=1e-6, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            warmup_factor = self.last_epoch / self.warmup_steps
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            progress = (self.last_epoch - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            cosine_factor = 0.5 * (1 + np.cos(np.pi * progress))
            return [self.eta_min + (base_lr - self.eta_min) * cosine_factor for base_lr in self.base_lrs]


class AudioDataset(Dataset):
    """Dataset for loading audio files from CSV"""
    def __init__(self, csv_path, sample_rate=24000, segment_duration=2.0, is_train=True):
        self.df = pd.read_csv(csv_path)
        self.sample_rate = sample_rate
        self.segment_duration = segment_duration
        self.segment_length = int(sample_rate * segment_duration)
        self.is_train = is_train
        
        # Filter out files that don't exist
        valid_files = []
        for idx, row in self.df.iterrows():
            if os.path.exists(row.iloc[0]):
                valid_files.append(row.iloc[0])
        self.audio_paths = valid_files
        print(f"Found {len(self.audio_paths)} valid audio files")
    
    def __len__(self):
        return len(self.audio_paths)
    
    def __getitem__(self, idx):
        audio_path = self.audio_paths[idx]
        
        try:
      
            audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
            
  =
            if len(audio) > self.segment_length:
                if self.is_train:
                    start = random.randint(0, len(audio) - self.segment_length)
                else:
                    start = 0 =
                audio = audio[start:start + self.segment_length]
            else:
                # Pad if too short
                audio = np.pad(audio, (0, self.segment_length - len(audio)))
     
            audio_tensor = torch.FloatTensor(audio).unsqueeze(0)
            
            return audio_tensor, audio_path
            
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            # Return silence if loading fails
            return torch.zeros(1, self.segment_length), audio_path


class BosonTrainer:
    def __init__(self, args):
        self.args = args
        self.distributed = False
        
        # Check if we're in a distributed environment
        if 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE']) > 1:
            self.distributed = True
            self.setup_ddp()
            self.device = torch.device(f'cuda:{args.local_rank}')
        else:
            # Single GPU mode
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            torch.cuda.set_device(0)
            set_random_seed(args.seed)
        
        # Load config
        with open(args.config, 'r') as f:
            self.config = json.load(f)
        
        # Initialize models
        self.model = self.build_model()
        self.discriminator = self.build_discriminator() if args.use_discriminator else None
        
        # Setup data loaders
        self.train_loader, self.val_loader = self.setup_data_loaders()
        
        # Setup optimizers
        self.optimizer_g = torch.optim.AdamW(
            self.model.parameters(),
            lr=args.learning_rate,
            betas=(0.5, 0.9),
            weight_decay=args.weight_decay
        )
        
        if self.discriminator is not None:
            self.optimizer_d = torch.optim.AdamW(
                self.discriminator.parameters(),
                lr=args.learning_rate * 2,  # Typically discriminator learns faster
                betas=(0.5, 0.9),
                weight_decay=args.weight_decay
            )
        
        # Initialize gradient scalers for mixed precision
        if args.use_mixed_precision:
            self.scaler_g = GradScaler()
            self.scaler_d = GradScaler() if self.discriminator is not None else None
        else:
            self.scaler_g = None
            self.scaler_d = None
        
        # Calculate total training steps
        self.total_steps = args.num_epochs * len(self.train_loader)
        
        # Setup schedulers with warmup
        self.scheduler_g = CosineWarmupScheduler(
            self.optimizer_g,
            warmup_steps=args.warmup_steps,
            total_steps=self.total_steps,
            eta_min=1e-6
        )
        
        if self.discriminator is not None:
            self.scheduler_d = CosineWarmupScheduler(
                self.optimizer_d,
                warmup_steps=args.warmup_steps,
                total_steps=self.total_steps,
                eta_min=1e-6
            )
        
        # Setup losses
        self.setup_losses()
        
        # Setup tensorboard
        if not self.distributed or rank() == 0:
            self.writer = SummaryWriter(
                log_dir=os.path.join(args.output_dir, 'logs', get_timestamp())
            )
        
        self.global_step = 0
        self.start_epoch = 0
        
        # Load checkpoint if exists
        if args.resume:
            self.load_checkpoint()
    
    def setup_ddp(self):
        """Initialize DDP"""
        if 'LOCAL_RANK' in os.environ:
            self.args.local_rank = int(os.environ['LOCAL_RANK'])
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(self.args.local_rank)
        set_random_seed(self.args.seed + rank())
    
    def build_model(self):
        """Build and wrap model with DDP if needed"""
        
        print(self.config)
        model = HiggsAudioTokenizer(
            n_filters=self.config['n_filters'],
            D=self.config['D'],
            target_bandwidths=self.config['target_bandwidths'],
            ratios=self.config['ratios'],
            sample_rate=self.config['sample_rate'],
            bins=self.config['bins'],
            n_q=self.config['n_q'],
            codebook_dim=self.config.get('codebook_dim', None),
            semantic_techer=self.config['semantic_techer'],
            device=self.device
        ).to(self.device)
        
        if self.distributed:
            # Broadcast model parameters to ensure all ranks have same initialization
            broadcast_tensors(model.parameters())
            # Wrap with DDP
            model = DDP(model, device_ids=[self.args.local_rank])
        
        return model
    
    # def build_discriminator(self):
    #     """Build discriminator with DDP if needed"""
    #     # Use sample rate from config
    #     discriminator = Discriminator(
    #         rates=[],  # No multi-rate discriminator for now
    #         periods=[2, 3, 5, 7, 11],
    #         fft_sizes=[2048, 1024, 512],
    #         sample_rate=self.config['sample_rate'],
    #     ).to(self.device)
        
    #     if self.distributed:
    #         broadcast_tensors(discriminator.parameters())
    #         discriminator = DDP(discriminator, device_ids=[self.args.local_rank])
        
    #     return discriminator
        
    def build_discriminator(self):
     
        discriminator = Discriminator(
            rates=[],  # No multi-rate discriminator
            periods=[2, 3, 5, 7, 11],
            fft_sizes=[2048, 1024, 512],
            sample_rate=self.config['sample_rate'],  # 24000
        ).to(self.device)
        
        if self.distributed:
            broadcast_tensors(discriminator.parameters())
            discriminator = DDP(discriminator, device_ids=[self.args.local_rank])
        
        return discriminator
    
    def setup_losses(self):

        # Basic losses
        self.l1_loss = L1Loss()
        self.stft_loss = MultiScaleSTFTLoss(
            window_lengths=[2048, 1024, 512, 256, 128],
            loss_fn=nn.L1Loss(),
            clamp_eps=1e-5,
            mag_weight=1.0,
            log_weight=1.0,
        )
        self.mel_loss = MelSpectrogramLoss(
            n_mels=[150, 80],
            window_lengths=[2048, 512],
            mel_fmin=[0.0, 0.0],
            mel_fmax=[None, None],
            clamp_eps=1e-5,
            mag_weight=1.0,
            log_weight=1.0,
        )
        
 
        if self.discriminator is not None:
            self.gan_loss = GANLoss(self.discriminator)
        
   
        self.loss_weights = {
            'rec': 1.,  # Waveform L1 loss
            'stft': 1.,  # Multi-scale STFT loss
            'mel': 45.0,  # Mel-spectrogram loss
            'commit': 0.25,  # Commitment loss
            'semantic': 1.,  # Semantic loss
            'gen': 1.,  # Generator adversarial loss
            'feat': 2.0,  # Feature matching loss
        }
    
    def setup_data_loaders(self):

        # Split data into train/val
        df = pd.read_csv(self.args.data_csv)
        n_total = len(df)
        n_train = int(n_total * 0.9)
        
        # Create temporary CSV files for train/val split
        train_csv = '/tmp/train_audio.csv'
        val_csv = '/tmp/val_audio.csv'
        
        if not self.distributed or rank() == 0:
            df[:n_train].to_csv(train_csv, index=False)
            df[n_train:].to_csv(val_csv, index=False)
        

        if self.distributed:
            dist.barrier()
        
        # Create datasets
        train_dataset = AudioDataset(
            train_csv,
            sample_rate=self.config['sample_rate'],
            segment_duration=self.args.segment_duration,
            is_train=True
        )
        
        val_dataset = AudioDataset(
            val_csv,
            sample_rate=self.config['sample_rate'],
            segment_duration=self.args.segment_duration,
            is_train=False
        )
        
        # Create samplers and loaders
        if self.distributed:
            train_sampler = DistributedSampler(train_dataset, shuffle=True)
            val_sampler = DistributedSampler(val_dataset, shuffle=False)
        else:
            train_sampler = None
            val_sampler = None
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            sampler=train_sampler,
            shuffle=(train_sampler is None),
            num_workers=self.args.num_workers,
            pin_memory=True,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.args.batch_size,
            sampler=val_sampler,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=True,
            drop_last=False
        )
        
        return train_loader, val_loader
    
    def is_main_process(self):
        """Check if this is the main process"""
        return not self.distributed or rank() == 0
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        if self.discriminator is not None:
            self.discriminator.train()
        
        if self.distributed:
            self.train_loader.sampler.set_epoch(epoch)
        
        total_losses = {
            'total': 0, 'rec': 0, 'stft': 0, 'mel': 0, 
            'commit': 0, 'semantic': 0, 'gen': 0, 'feat': 0, 'disc': 0
        }
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}', disable=not self.is_main_process())
        
        for batch_idx, (audio, paths) in enumerate(pbar):
            audio = audio.to(self.device)
            
            # Create AudioSignal objects for loss computation
            audio_signal = AudioSignal(audio, self.config['sample_rate'])
            
            # Forward pass with random bandwidth
            bw_idx = random.randint(0, len(self.config['target_bandwidths']) - 1)
            bw = self.config['target_bandwidths'][bw_idx]
            
            # Use autocast for mixed precision
            with autocast(dtype=torch.bfloat16, enabled=self.args.use_mixed_precision):
                output, commit_loss, semantic_loss, _ = self.model(audio, bw)
                recons_signal = AudioSignal(output, self.config['sample_rate'])
            

            use_discriminator = (self.discriminator is not None and 
                                self.global_step >= self.args.discriminator_start_step)
            

            if use_discriminator and self.global_step % self.args.disc_interval == 0:
                self.optimizer_d.zero_grad()
                
                with autocast(dtype=torch.bfloat16, enabled=self.args.use_mixed_precision):
                    disc_loss = self.gan_loss.discriminator_loss(recons_signal, audio_signal)
                
                if self.scaler_d is not None:
                    self.scaler_d.scale(disc_loss).backward()
                    self.scaler_d.unscale_(self.optimizer_d)
                    torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 10.0)
                    self.scaler_d.step(self.optimizer_d)
                    self.scaler_d.update()
                else:
                    disc_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 10.0)
                    self.optimizer_d.step()
                
                self.scheduler_d.step()
                total_losses['disc'] += disc_loss.item()
            
            # Train generator
            losses = {}
            
            # Compute losses with autocast
            with autocast(dtype=torch.bfloat16, enabled=self.args.use_mixed_precision):
                # Reconstruction losses
                losses['rec'] = self.l1_loss(recons_signal, audio_signal)
                losses['stft'] = self.stft_loss(recons_signal, audio_signal)
                losses['mel'] = self.mel_loss(recons_signal, audio_signal)
                # losses['mel'] = torch.tensor(0.0, device=self.device) # uncomment this for the first 30k steps, it's faster if you pretrain it on semantic / commit loss first
                losses['commit'] = commit_loss
                losses['semantic'] = semantic_loss
                
                # GAN losses if discriminator is active
                if use_discriminator:
                    gen_loss, feat_loss = self.gan_loss.generator_loss(recons_signal, audio_signal)
                    losses['gen'] = gen_loss
                    losses['feat'] = feat_loss
                else:
                    # Set to zero for logging purposes
                    losses['gen'] = torch.tensor(0.0, device=self.device)
                    losses['feat'] = torch.tensor(0.0, device=self.device)
                
                # Total weighted loss
                total_loss = sum(self.loss_weights.get(k, 0) * v for k, v in losses.items() 
                               if k not in ['gen', 'feat'] or use_discriminator)
            
            # Backward pass
            self.optimizer_g.zero_grad()
            
            if self.scaler_g is not None:
                self.scaler_g.scale(total_loss).backward()
                self.scaler_g.unscale_(self.optimizer_g)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler_g.step(self.optimizer_g)
                self.scaler_g.update()
            else:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer_g.step()
            
            self.scheduler_g.step()
            
            # Update metrics
            total_losses['total'] += total_loss.item()
            for k, v in losses.items():
                total_losses[k] += v.item()
            
            # Update progress bar
            if self.is_main_process():
                pbar.set_postfix({
                    'loss': f'{total_loss.item():.4f}',
                    'rec': f'{losses["rec"].item():.4f}',
                    'mel': f'{losses["mel"].item():.4f}',
                    'commit_loss': f'{losses["commit"].item():.4f}',
                    'semantic_loss': f'{losses["semantic"].item():.4f}',
                    'lr': f'{self.scheduler_g.get_last_lr()[0]:.9f}',
                    'disc': 'ON' if use_discriminator else 'OFF',
                    'step': self.global_step
                })
            
            # Log to tensorboard
            if self.is_main_process() and self.global_step % self.args.log_interval == 0:
                for k, v in losses.items():
                    self.writer.add_scalar(f'train/{k}_loss', v.item(), self.global_step)
                self.writer.add_scalar('train/total_loss', total_loss.item(), self.global_step)
                self.writer.add_scalar('train/lr', self.scheduler_g.get_last_lr()[0], self.global_step)
                self.writer.add_scalar('train/bandwidth', bw, self.global_step)
                self.writer.add_scalar('train/discriminator_active', float(use_discriminator), self.global_step)
                if use_discriminator:
                    self.writer.add_scalar('train/disc_loss', total_losses['disc'] / max(1, batch_idx), self.global_step)
                if self.scaler_g is not None:
                    self.writer.add_scalar('train/grad_scale', self.scaler_g.get_scale(), self.global_step)
            
            # Save checkpoint at step intervals
            if self.global_step > 0 and self.global_step % self.args.save_step_interval == 0:
                self.save_checkpoint_step(self.global_step)
                if self.is_main_process():
                    print(f"\nSaved checkpoint at step {self.global_step}")
            
            self.global_step += 1
        
        # Return average losses
        n_batches = len(self.train_loader)
        return {k: v / n_batches for k, v in total_losses.items()}
    
    @torch.no_grad()
    def validate(self, epoch):
        """Validation loop"""
        self.model.eval()
        
        total_losses = {
            'total': 0, 'rec': 0, 'stft': 0, 'mel': 0, 
            'commit': 0, 'semantic': 0
        }
        

        audio_samples = {'train': [], 'val': []}
        
        for batch_idx, (audio, paths) in enumerate(tqdm(self.val_loader, desc='Validation', disable=not self.is_main_process())):
            audio = audio.to(self.device)
            audio_signal = AudioSignal(audio, self.config['sample_rate'])
            
            # Use medium bandwidth for validation
            bw = self.config['target_bandwidths'][2]
            
            # Use autocast for validation too
            with autocast(dtype=torch.bfloat16, enabled=self.args.use_mixed_precision):
                output, commit_loss, semantic_loss, _ = self.model(audio, bw)
                recons_signal = AudioSignal(output, self.config['sample_rate'])
                
                # Compute losses
                losses = {
                    'rec': self.l1_loss(recons_signal, audio_signal),
                    'stft': self.stft_loss(recons_signal, audio_signal),
                    'mel': self.mel_loss(recons_signal, audio_signal),
                    'commit': commit_loss,
                    'semantic': semantic_loss
                }
                
                total_loss = sum(self.loss_weights.get(k, 0) * v for k, v in losses.items())
            
            total_losses['total'] += total_loss.item()
            for k, v in losses.items():
                total_losses[k] += v.item()
            
            # Collect audio samples for tensorboard (first 3 from validation)
            if self.is_main_process() and len(audio_samples['val']) < 3:
                audio_samples['val'].append({
                    'original': audio[0].cpu(),
                    'reconstructed': output[0].cpu(),
                    'path': paths[0]
                })
        
        # Get train samples for comparison
        if self.is_main_process():
            self.model.eval()
            for batch_idx, (audio, paths) in enumerate(self.train_loader):
                if len(audio_samples['train']) >= 3:
                    break
                audio = audio.to(self.device)
                bw = self.config['target_bandwidths'][2]
                with autocast(dtype=torch.bfloat16, enabled=self.args.use_mixed_precision):
                    output, _, _, _ = self.model(audio, bw)
                audio_samples['train'].append({
                    'original': audio[0].cpu(),
                    'reconstructed': output[0].cpu(),
                    'path': paths[0]
                })
        
        # Log audio samples to tensorboard
        if self.is_main_process():
            for split in ['train', 'val']:
                for idx, sample in enumerate(audio_samples[split]):
                    self.writer.add_audio(
                        f'{split}/original_{idx}',
                        sample['original'],
                        epoch,
                        sample_rate=self.config['sample_rate']
                    )
                    self.writer.add_audio(
                        f'{split}/reconstructed_{idx}',
                        sample['reconstructed'],
                        epoch,
                        sample_rate=self.config['sample_rate']
                    )
        
        # Average losses
        n_batches = len(self.val_loader)
        val_metrics = {k: v / n_batches for k, v in total_losses.items()}
        
        # Log validation metrics
        if self.is_main_process():
            for key, value in val_metrics.items():
                self.writer.add_scalar(f'val/{key}_loss', value, epoch)
        
        return val_metrics
        
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint (epoch-based)"""
        if not self.is_main_process():
            return
        
        model_state = self.model.module.state_dict() if self.distributed else self.model.state_dict()
        
        # Get current learning rates for verification
        current_lr_g = self.scheduler_g.get_last_lr()[0]
        
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': model_state,
            'optimizer_g_state_dict': self.optimizer_g.state_dict(),
            'scheduler_g_state_dict': self.scheduler_g.state_dict(),
            'scheduler_g_last_epoch': self.scheduler_g.last_epoch,  # Explicitly save this
            'current_lr_g': current_lr_g,  # Save for verification
            'config': self.config,
            'args': self.args
        }
        
        # Save gradient scaler states if using mixed precision
        if self.scaler_g is not None:
            checkpoint['scaler_g_state_dict'] = self.scaler_g.state_dict()
        
        if self.discriminator is not None:
            disc_state = self.discriminator.module.state_dict() if self.distributed else self.discriminator.state_dict()
            current_lr_d = self.scheduler_d.get_last_lr()[0]
            checkpoint['discriminator_state_dict'] = disc_state
            checkpoint['optimizer_d_state_dict'] = self.optimizer_d.state_dict()
            checkpoint['scheduler_d_state_dict'] = self.scheduler_d.state_dict()
            checkpoint['scheduler_d_last_epoch'] = self.scheduler_d.last_epoch
            checkpoint['current_lr_d'] = current_lr_d
            
            if self.scaler_d is not None:
                checkpoint['scaler_d_state_dict'] = self.scaler_d.state_dict()
        
        # Save latest checkpoint
        checkpoint_path = os.path.join(self.args.output_dir, 'checkpoints', 'latest.pth')
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.args.output_dir, 'checkpoints', 'best.pth')
            torch.save(checkpoint, best_path)
        
        # Save periodic checkpoint
        if epoch % self.args.save_interval == 0:
            epoch_path = os.path.join(self.args.output_dir, 'checkpoints', f'epoch_{epoch}.pth')
            torch.save(checkpoint, epoch_path)


    def save_checkpoint_step(self, step):
        """Save model checkpoint (step-based)"""
        if not self.is_main_process():
            return
        
        # Get current epoch from training loop
        current_epoch = step // len(self.train_loader)
        
        model_state = self.model.module.state_dict() if self.distributed else self.model.state_dict()
        
        # Get current learning rates for verification
        current_lr_g = self.scheduler_g.get_last_lr()[0]
        
        checkpoint = {
            'epoch': current_epoch,
            'global_step': step,
            'model_state_dict': model_state,
            'optimizer_g_state_dict': self.optimizer_g.state_dict(),
            'scheduler_g_state_dict': self.scheduler_g.state_dict(),
            'scheduler_g_last_epoch': self.scheduler_g.last_epoch,  # Explicitly save this
            'current_lr_g': current_lr_g,  # Save for verification
            'config': self.config,
            'args': self.args
        }
        
        # Save gradient scaler states if using mixed precision
        if self.scaler_g is not None:
            checkpoint['scaler_g_state_dict'] = self.scaler_g.state_dict()
        
        if self.discriminator is not None:
            disc_state = self.discriminator.module.state_dict() if self.distributed else self.discriminator.state_dict()
            current_lr_d = self.scheduler_d.get_last_lr()[0]
            checkpoint['discriminator_state_dict'] = disc_state
            checkpoint['optimizer_d_state_dict'] = self.optimizer_d.state_dict()
            checkpoint['scheduler_d_state_dict'] = self.scheduler_d.state_dict()
            checkpoint['scheduler_d_last_epoch'] = self.scheduler_d.last_epoch
            checkpoint['current_lr_d'] = current_lr_d
            
            if self.scaler_d is not None:
                checkpoint['scaler_d_state_dict'] = self.scaler_d.state_dict()
        
        # Create checkpoint directory if it doesn't exist
        checkpoint_dir = os.path.join(self.args.output_dir, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save step-based checkpoint
        step_path = os.path.join(self.args.output_dir, 'checkpoints', f'step_{step}.pth')
        torch.save(checkpoint, step_path)
        
        # Also update latest checkpoint
        latest_path = os.path.join(self.args.output_dir, 'checkpoints', 'latest.pth')
        torch.save(checkpoint, latest_path)
        
        # Keep only the last N step-based checkpoints to save disk space
        if self.args.keep_last_n_steps > 0:
            checkpoint_dir = os.path.join(self.args.output_dir, 'checkpoints')
            step_checkpoints = sorted([f for f in os.listdir(checkpoint_dir) if f.startswith('step_')])
            if len(step_checkpoints) > self.args.keep_last_n_steps:
                for old_checkpoint in step_checkpoints[:-self.args.keep_last_n_steps]:
                    os.remove(os.path.join(checkpoint_dir, old_checkpoint))


    def load_checkpoint(self):

        checkpoint_path = os.path.join(self.args.output_dir, 'checkpoints', 'latest.pth')
        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            
            # Load model state
            if self.distributed:
                self.model.module.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Load optimizer state
            self.optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
            
            # Load scheduler state
            self.scheduler_g.load_state_dict(checkpoint['scheduler_g_state_dict'])
            
            # Restore scheduler's last_epoch from checkpoint
            if 'scheduler_g_last_epoch' in checkpoint:
                self.scheduler_g.last_epoch = checkpoint['scheduler_g_last_epoch']
            else:
             
                self.scheduler_g.last_epoch = checkpoint['global_step']
            
            # Force scheduler to recompute its internal state
            self.scheduler_g._last_lr = self.scheduler_g.get_lr()
            
            # Load gradient scaler state if using mixed precision
            if self.scaler_g is not None and 'scaler_g_state_dict' in checkpoint:
                self.scaler_g.load_state_dict(checkpoint['scaler_g_state_dict'])
            
            # Load discriminator if present
            if self.discriminator is not None and 'discriminator_state_dict' in checkpoint:
                if self.distributed:
                    self.discriminator.module.load_state_dict(checkpoint['discriminator_state_dict'])
                else:
                    self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
                self.optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
                self.scheduler_d.load_state_dict(checkpoint['scheduler_d_state_dict'])
                
                # Restore discriminator scheduler's last_epoch
                if 'scheduler_d_last_epoch' in checkpoint:
                    self.scheduler_d.last_epoch = checkpoint['scheduler_d_last_epoch']
                else:
                    self.scheduler_d.last_epoch = checkpoint['global_step']
                
                self.scheduler_d._last_lr = self.scheduler_d.get_lr()
                
            
                if self.scaler_d is not None and 'scaler_d_state_dict' in checkpoint:
                    self.scaler_d.load_state_dict(checkpoint['scaler_d_state_dict'])
            
            # Restore training state
            self.start_epoch = checkpoint['epoch'] + 1
            self.global_step = checkpoint['global_step']
            
            # Verify learning rate restoration
            current_lr_g = self.scheduler_g.get_last_lr()[0]
            saved_lr_g = checkpoint.get('current_lr_g', None)
            
            print(f"\n{'='*60}")
            print(f"CHECKPOINT LOADED SUCCESSFULLY")
            print(f"{'='*60}")
            print(f"Resumed from epoch: {checkpoint['epoch']}")
            print(f"Global step: {self.global_step}")
            print(f"Scheduler last_epoch: {self.scheduler_g.last_epoch}")
            print(f"Current learning rate (generator): {current_lr_g:.9f}")
            print(f"Mixed precision: {'ENABLED' if self.args.use_mixed_precision else 'DISABLED'}")
            if saved_lr_g is not None:
                print(f"Saved learning rate (generator): {saved_lr_g:.9f}")
                if abs(current_lr_g - saved_lr_g) > 1e-9:
                    print("⚠️  WARNING: Learning rate mismatch! This might indicate improper state restoration.")
            
            if self.discriminator is not None:
                current_lr_d = self.scheduler_d.get_last_lr()[0]
                saved_lr_d = checkpoint.get('current_lr_d', None)
                print(f"Current learning rate (discriminator): {current_lr_d:.9f}")
                if saved_lr_d is not None:
                    print(f"Saved learning rate (discriminator): {saved_lr_d:.9f}")
                print(f"Discriminator status: {'ACTIVE' if self.global_step >= self.args.discriminator_start_step else f'INACTIVE (starts at step {self.args.discriminator_start_step})'}")
            
            print(f"Next epoch: {self.start_epoch}")
            print(f"Next step checkpoint at: step {((self.global_step // self.args.save_step_interval) + 1) * self.args.save_step_interval}")
            print(f"{'='*60}\n")
            
  g
            if self.global_step > 0:
                temp_scheduler = CosineWarmupScheduler(
                    self.optimizer_g, 
                    self.args.warmup_steps, 
                    self.total_steps,
                    eta_min=1e-6,
                    last_epoch=-1
                )
                # Step it to the current global step
                for _ in range(self.global_step):
                    temp_scheduler.step()
                expected_lr = temp_scheduler.get_last_lr()[0]
                if abs(current_lr_g - expected_lr) > 1e-9:
                    print(f"⚠️  Learning rate verification failed!")
                    print(f"   Expected: {expected_lr:.9f}")
                    print(f"   Got: {current_lr_g:.9f}")
                    print("   The scheduler state might not be properly restored.")
        else:
            print(f"No checkpoint found at {checkpoint_path}, starting from scratch")
        
    def train(self):
        """Main training loop"""
        best_val_loss = float('inf')
        
        # Print training configuration
        if self.is_main_process():
            print(f"\n{'='*50}")
            print(f"Training Configuration:")
            print(f"{'='*50}")
            print(f"Total epochs: {self.args.num_epochs}")
            print(f"Steps per epoch: {len(self.train_loader)}")
            print(f"Total steps: {self.total_steps}")
            print(f"Warmup steps: {self.args.warmup_steps}")
            print(f"Mixed precision training: {'ENABLED (bfloat16)' if self.args.use_mixed_precision else 'DISABLED'}")
            print(f"Discriminator starts at step: {self.args.discriminator_start_step}")
            print(f"Checkpoint saving:")
            print(f"  - Every {self.args.save_interval} epochs")
            print(f"  - Every {self.args.save_step_interval} steps")
            print(f"  - Keep last {self.args.keep_last_n_steps} step checkpoints")
            if self.start_epoch > 0:
                print(f"RESUMING from epoch {self.start_epoch}, step {self.global_step}")
            print(f"{'='*50}\n")
        
        for epoch in range(self.start_epoch, self.args.num_epochs):
            # IMPORTANT: Set the epoch for distributed sampler when resuming
            # This ensures proper data shuffling across epochs
            if self.distributed and hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(epoch)
            
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate(epoch)
            
            # Log epoch metrics
            if self.is_main_process():
                print(f"\nEpoch {epoch} Summary:")
                print(f"Train - Total: {train_metrics['total']:.4f}, Rec: {train_metrics['rec']:.4f}, "
                    f"STFT: {train_metrics['stft']:.4f}, Mel: {train_metrics['mel']:.4f}, "
                    f"Commit: {train_metrics['commit']:.4f}, Semantic: {train_metrics['semantic']:.4f}")
                if self.discriminator is not None:
                    print(f"       Gen: {train_metrics['gen']:.4f}, Feat: {train_metrics['feat']:.4f}, "
                        f"Disc: {train_metrics['disc']:.4f}")
                    print(f"       Discriminator Status: {'Active' if self.global_step >= self.args.discriminator_start_step else f'Starting at step {self.args.discriminator_start_step}'}")
                print(f"Val   - Total: {val_metrics['total']:.4f}, Rec: {val_metrics['rec']:.4f}, "
                    f"STFT: {val_metrics['stft']:.4f}, Mel: {val_metrics['mel']:.4f}, "
                    f"Commit: {val_metrics['commit']:.4f}, Semantic: {val_metrics['semantic']:.4f}")
                print(f"Current Step: {self.global_step}, Next step checkpoint at: {((self.global_step // self.args.save_step_interval) + 1) * self.args.save_step_interval}")
                print(f"Current LR: {self.scheduler_g.get_last_lr()[0]:.9f}")
            
            # Save checkpoint
            is_best = val_metrics['total'] < best_val_loss
            if is_best:
                best_val_loss = val_metrics['total']
            self.save_checkpoint(epoch, is_best)
        
        # Save final model
        if self.is_main_process():
            model_state = self.model.module.state_dict() if self.distributed else self.model.state_dict()
            
            final_path = os.path.join(self.args.output_dir, 'checkpoints', 'final.pth')
            torch.save({
                'model_state_dict': model_state,
                'config': self.config
            }, final_path)
            
            # Also save just the model weights in the format expected by the original code
            model_only_path = os.path.join(self.args.output_dir, 'model.pth')
            torch.save(model_state, model_only_path)
            
            # Copy config
            import shutil
            shutil.copy(self.args.config, os.path.join(self.args.output_dir, 'config.json'))
        
        # Cleanup
        if self.is_main_process():
            self.writer.close()
        if self.distributed:
            dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description='Train Boson Audio Codec')
    
    # Data arguments
    parser.add_argument('--data_csv', type=str, required=True,
                        help='Path to CSV file containing audio file paths')
    parser.add_argument('--config', type=str, default='config.json',
                        help='Path to config JSON file')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=28,
                        help='Batch size per GPU')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--segment_duration', type=float, default=2.,
                        help='Audio segment duration in seconds')
    
    # Mixed precision training
    parser.add_argument('--use_mixed_precision', action='store_true',
                        help='Use bfloat16 mixed precision training')
    
    # Scheduler arguments
    parser.add_argument('--warmup_steps', type=int, default=5000,
                        help='Number of warmup steps for cosine scheduler')
    
    # Loss arguments
    parser.add_argument('--use_discriminator', action='store_true',
                        help='Use adversarial training with discriminator')
    parser.add_argument('--discriminator_start_step', type=int, default=30_000,
                        help='Start training discriminator after N steps')
    parser.add_argument('--disc_interval', type=int, default=1,
                        help='Train discriminator every N steps')
    
    # System arguments
    parser.add_argument('--output_dir', type=str, default='outputs_mp_cqt',
                        help='Output directory for checkpoints and logs')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--local_rank', type=int, default=0,
                        help='Local rank for distributed training')
    
    # Logging arguments
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Log every N steps')
    parser.add_argument('--save_interval', type=int, default=1,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--save_step_interval', type=int, default=1000,
                        help='Save checkpoint every N steps')
    parser.add_argument('--keep_last_n_steps', type=int, default=5,
                        help='Keep only the last N step-based checkpoints (0 to keep all)')
    
    # Resume training
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from latest checkpoint') # NOTE: you gotta change your desired checkpoint's name to latest.pth
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Train
    trainer = BosonTrainer(args)
    trainer.train()


if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')
    main()