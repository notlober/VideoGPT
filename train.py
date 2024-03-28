import json
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

# Custom dataset class for video captions
class VideoCaptionsDataset(Dataset):
    def __init__(self, json_file, video_prefix='video'):
        # Load the JSON file
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        
        # Store the video prefix
        self.video_prefix = video_prefix

    def __len__(self):
        # Return the number of videos
        return len(self.data)

    def __getitem__(self, idx):
        # Get the video ID
        video_id = f'{self.video_prefix}{idx}'
        
        # Get the captions for the video
        captions = self.data[str(idx)]['captions']
        
        # Return the video ID and captions as a tuple
        return video_id, captions

# Path to your JSON file
json_file = '/kaggle/input/msrvttcaptiondataset/Result/caption.json'

# Create the dataset
dataset = VideoCaptionsDataset(json_file)

# Create a DataLoader
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Example usage
for video_id, captions in dataloader:
    print(f'Video ID: {video_id}')
    print(f'Captions: {captions}')
    break

import json
import os

from datasets import Dataset, load_from_disk
import numpy as np
import torch
from torch.cuda.amp import GradScaler
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import VisionEncoderDecoderModel, AutoImageProcessor, AutoTokenizer, default_data_collator, get_scheduler

device = torch.device("cuda")

# MODEL
encoder = "facebook/timesformer-base-finetuned-k600"
decoder = "openai-community/gpt2"

image_processor = AutoImageProcessor.from_pretrained("facebook/timesformer-base-finetuned-k600")
tokenizer = AutoTokenizer.from_pretrained(decoder)
tokenizer.pad_token = tokenizer.eos_token

model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(encoder, decoder)
model = model.to(device)

model.config.decoder_start_token_id = tokenizer.bos_token_id
model.config.pad_token_id = tokenizer.pad_token_id
model.config.eos_token_id = tokenizer.eos_token_id

model.config.max_length = 64
model.config.num_beams = 8
model.config.early_stopping = True

import cv2
import numpy as np

def load_video_frames(video_path, num_frames=8):
    """
    Load a fixed number of frames from a video file.

    Args:
        video_path (str): Path to the video file.
        num_frames (int): Number of frames to load.

    Returns:
        np.ndarray: Array of video frames.
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    frames = []
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        else:
            break

    cap.release()
    return np.array(frames)

print(model)

import os

# Set up training parameters
EPOCHS = 3
optimizer = AdamW(model.parameters(), lr=1e-4)
training_steps = EPOCHS * len(dataset)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=2000,
    num_training_steps=training_steps,
)

# Define the base directory where your video files are located
video_base_path = "/kaggle/input/msrvtt/TrainValVideo"

# Training loop
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{EPOCHS}")

    for video_id, captions in progress_bar:
        optimizer.zero_grad()

        # Extract the first element from the tuple
        video_id = video_id[0]
        captions = [caption[0] for caption in captions]

        # Construct the full path to the video file
        video_path = os.path.join(video_base_path, f"{video_id}.mp4")  # Adjust the file extension if necessary
            
        # Load and process video frames and captions
        video_frames = load_video_frames(video_path)
        video_frames = [frame for frame in video_frames]  # Ensure video_frames is a list of frames
        video_frames = image_processor(images=video_frames, return_tensors="pt").pixel_values.to(device)
        captions = tokenizer(captions, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)
        labels = captions[:, 1:].clone()  # Shift captions to the right
        labels[captions[:, 1:] == tokenizer.pad_token_id] = -100  # Ignore pad tokens in the loss

        # Forward pass
        outputs = model(pixel_values=video_frames, labels=labels)
        loss = outputs.loss
        train_loss += loss.item()
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        progress_bar.set_postfix({"Loss": f"{train_loss / (progress_bar.n + 1):.4f}"})
        
    print(f"Epoch {epoch + 1} finished with average loss: {train_loss / len(dataloader):.4f}")

# Save model and tokenizer
model.save_pretrained("videogpt")
tokenizer.save_pretrained("videogpt")
image_processor.save_pretrained("videogpt")
