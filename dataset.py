import torch
from torchvision.io import read_video
import torchvision
import av
import warnings
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
warnings.filterwarnings("ignore")
def extract_frames(video_tensor, num_frames):
    total_frames = video_tensor.shape[0]
    
    if num_frames == total_frames:
        return video_tensor  # Return all frames if requested number is greater or equal to total frames
    elif num_frames > total_frames:
        last_frame = video_tensor[-1:]  # Select the last frame
        num_repeats = num_frames - total_frames
        repeated_frames = last_frame.repeat(num_repeats, 1, 1, 1)  # Repeat the last frame to match num_frames
        selected_frames = torch.cat([video_tensor, repeated_frames], dim=0)  # Concatenate the repeated frames
        return selected_frames
    else:
        # Calculate indices for equally spaced frames
        indices = torch.linspace(0, total_frames - 1, num_frames).round().long()
        
        # Extract frames based on the calculated indices
        selected_frames = video_tensor[indices]
        
        return selected_frames


class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, num_frames, type, transform=None):
        self.transform = transform
        self.num_frames = num_frames
        self.dataframe = dataframe[dataframe['type'] == type].reset_index(drop=True) 
        self.dataframe['index_label'] = label_encoder.fit_transform(self.dataframe['label'])
    def __len__(self):
        return len(self.dataframe)
    def __getitem__(self, idx):
        video, audio, info = read_video(self.dataframe['video_path'][idx], pts_unit="sec")
        video = extract_frames(video, self.num_frames)
        if video.shape[0] < self.num_frames:
            print(video.shape[0])
        if self.transform is not None:
            video = self.transform(video)
        label = self.dataframe['index_label'][idx]
        return video, label