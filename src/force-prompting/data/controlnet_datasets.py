import os
import glob
import random


import torch
import math
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
from PIL import Image
from decord import VideoReader
from torch.utils.data.dataset import Dataset
from controlnet_aux import CannyDetector, HEDdetector

def unpack_mm_params(p):
    if isinstance(p, (tuple, list)):
        return p[0], p[1]
    elif isinstance(p, (int, float)):
        return p, p
    raise Exception(f'Unknown input parameter type.\nParameter: {p}.\nType: {type(p)}')


def resize_for_crop(image, min_h, min_w):
    img_h, img_w = image.shape[-2:]
    
    # Calculate the scaling coefficients
    h_coef = min_h / img_h
    w_coef = min_w / img_w
    
    if img_h >= min_h and img_w >= min_w:
        # Both dimensions are larger, scale down to the minimum required size
        coef = min(h_coef, w_coef)
    elif img_h <= min_h and img_w <= min_w:
        # Both dimensions are smaller, scale up to the minimum required size
        coef = max(h_coef, w_coef)
    else:
        # Mixed case - one dimension is larger, one is smaller
        # Scale up to ensure both dimensions meet the minimum
        coef = max(h_coef, w_coef)
    
    # Calculate new dimensions
    out_h = int(img_h * coef)
    out_w = int(img_w * coef)
    
    # Ensure dimensions are at least the minimum required
    # This handles cases where rounding down during int conversion drops below minimum
    if out_h < min_h:
        out_h = min_h
    if out_w < min_w:
        out_w = min_w
    
    resized_image = transforms.functional.resize(image, (out_h, out_w), antialias=True)
    return resized_image



class BaseClass(Dataset):
    def __init__(
            self, 
            video_root_dir,
            image_size=(320, 512), 
            stride=(1, 2), 
            sample_n_frames=25,
            controlnet_type='',
        ):
        self.height, self.width = unpack_mm_params(image_size)
        self.stride_min, self.stride_max = unpack_mm_params(stride)
        self.video_root_dir = video_root_dir
        self.sample_n_frames = sample_n_frames
        
        self.length = 0
        
        self.controlnet_type = controlnet_type

    def load_pixel_values_image(self, image_path):

        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = image.resize((self.width, self.height), Image.LANCZOS)
        np_image = np.array(image) # (480, 720, 3)
        pixel_values = torch.from_numpy(np_image).permute(2, 0, 1).contiguous().unsqueeze(0) # (1, 3, 480, 720)
        pixel_values = pixel_values / 127.5 - 1

        return pixel_values
        
    def __len__(self):
        return self.length
        
    def get_batch(self, idx):
        raise Exception('Get batch method is not realized.')

    def __getitem__(self, idx):
        raise Exception('Get item method is not realized.')



class ForcePromptingDataset_PointForce(BaseClass):
    def __init__(self, csv_path, is_validation_dataset=False, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.is_validation_dataset = is_validation_dataset

        if is_validation_dataset:
            self.media_type = "image"
            blob_ext =  "*.png"
        else:
            self.media_type = "video"
            blob_ext = "*.mp4"
            
        file_paths = glob.glob(os.path.join(self.video_root_dir, blob_ext))
        file_names = set([os.path.basename(x) for x in file_paths]) # list of videos or images...
        self.df = pd.read_csv(csv_path)

        # only keep the rows in the csv whose videos we can find
        self.df['checked'] = self.df[self.media_type].map(lambda x, files=file_names: int(x in files))
        self.df = self.df[self.df['checked'] == True]

        self.min_force = float(self.df["force"].min())
        self.max_force = float(self.df["force"].max())

        self.length = self.df.shape[0]

    def get_batch(self, idx):

        item = self.df.iloc[idx]
        caption = item['caption']
        file_name = item[self.media_type]
        force = item['force']
        angle = item['angle']
        file_path = os.path.join(self.video_root_dir, file_name)

        if self.media_type == "image":
            pixel_values = self.load_pixel_values_image(file_path) # (1, 3, 480, 720) of torch.float32 in [-1, 1]
            file_id = file_name.split(".png")[0]
            x_pos = item["coordx"] / item["width"]
            y_pos = item["coordy"] / item["height"]

        elif self.media_type == "video":
            pixel_values = self.load_pixel_values_video(file_path) # (49, 3, 480, 720) of torch.float32 in [-1, 1]
            # tensor_to_video_ffmpeg(0.5 + pixel_values/2, "pixel_values.mp4", fps=10)
            file_id = file_name.split(".mp4")[0]

            # AUTOMATIC RAMDOM CROPPING PROCEDURE, but only for the carnation...
            if file_id.startswith("carnation"):
                crop_zoom_amount =  np.random.uniform(1.0, 1.3) # 1.0 means no zoom; 1.3 means zoom in 1.3x
                new_width = int(item["width"] / crop_zoom_amount) - int(item["width"] / crop_zoom_amount) % 2
                new_height = int(item["height"] / crop_zoom_amount) - int(item["height"] / crop_zoom_amount) % 2

                num_tries = 0
                while num_tries < 100:
                    new_origin_x_pos = int(np.random.uniform(0, item["width"] - item["width"]/crop_zoom_amount))
                    new_origin_y_pos = int(np.random.uniform(0, item["height"] - item["height"]/crop_zoom_amount))

                    if item["coordx"] in range(new_origin_x_pos + 50, new_origin_x_pos+new_width - 50) and item["coordy"] in range(new_origin_y_pos + 50, new_origin_y_pos+new_height - 50):
                        num_tries = 100
                    num_tries += 1
                pixel_values = pixel_values[:, :, item["height"] - (new_origin_y_pos+new_height):item["height"] - new_origin_y_pos, new_origin_x_pos:new_origin_x_pos+new_width]
                pixel_values = resize_for_crop(pixel_values, self.height, self.width) # (49, 3, 480, 720)

                # tensor_to_video_ffmpeg(0.5 + new_pixel_values/2, "pixel_values_new.mp4", fps=10)
                new_x_pos = (new_width / item["width"]) * (item["coordx"] + new_origin_x_pos)
                new_y_pos = (new_height / item["height"]) * (item["coordy"] - new_origin_y_pos)

                new_r = (crop_zoom_amount / 2) * math.sqrt((item["coordx"] - new_origin_x_pos)**2 + (item["coordy"] - new_origin_y_pos)**2)
                new_theta = math.atan((item["coordy"] - new_origin_y_pos) / (item["coordx"] - new_origin_x_pos))

                new_x_pos = int(new_r * math.cos(new_theta))
                new_y_pos = self.height - int(new_r * math.sin(new_theta))

                x_pos = new_x_pos / self.width
                y_pos = 1 - new_y_pos / self.height
            
            else:
                x_pos = item["coordx"] / item["width"]
                y_pos = item["coordy"] / item["height"]

            # new_pixel_values_with_blob = torch.clip(new_pixel_values + 10* self.get_gaussian_blob(x=new_x_pos, y=new_y_pos, radius=10, amplitude=1.0, shape=(3, 480, 720)), max=1.0)
            # tensor_to_video_ffmpeg(0.5 + new_pixel_values_with_blob/2, "pixel_values_new_with_blob.mp4", fps=10)

        controlnet_signal = self.load_controlnet_signal(
            force, angle, x_pos, y_pos,
        )

        return pixel_values, caption, controlnet_signal, force, angle, x_pos, y_pos, file_id

    def __getitem__(self, idx):
        while True:
            try:
                pixel_values, caption, controlnet_signal, force, angle, x_pos, y_pos, file_id = self.get_batch(idx)
                # video, caption, controlnet_video = self.get_batch(idx)
                break
            except Exception as e:
                print("EXCEPTION HERE!!!", e) # this prints 'text' incessantly
                idx = random.randint(0, self.length - 1)
            
        pixel_values = [
            resize_for_crop(x, self.height, self.width) for x in [pixel_values]
        ][0]
        pixel_values = [
            transforms.functional.center_crop(x, (self.height, self.width)) for x in [pixel_values]
        ][0]
        data = {
            'file_id' : file_id,
            'video': pixel_values, 
            'caption': caption, 
            'controlnet_video': controlnet_signal,
            'force': force,
            'angle': angle,
            'x_pos': x_pos,
            'y_pos': y_pos,
        }
        return data

    def load_pixel_values_video(self, video_path):

        video_reader = VideoReader(video_path)

        if "carnation" in video_path:
            indices = np.array([2*i for i in range(self.sample_n_frames)], dtype=int)
        else:
            indices = np.array([i for i in range(10, 10 + self.sample_n_frames)], dtype=int)
        
        # Get the selected frames
        np_video = video_reader.get_batch(indices).asnumpy() # (49, 960, 1440, 3)
        pixel_values = torch.from_numpy(np_video).permute(0, 3, 1, 2).contiguous() # (49, 3, 960, 1440) of uint8 in [0, 255]
        pixel_values = pixel_values / 127.5 - 1 # (49, 3, 960, 1440) of torch.float32 in [-1, 1]
        del video_reader

        return pixel_values


    def get_gaussian_blob(self, x, y, radius=10, amplitude=1.0, shape=(3, 480, 720), device=None):
        """
        Create a tensor containing a Gaussian blob at the specified location.
        
        Args:
            x (int): x-coordinate of the blob center
            y (int): y-coordinate of the blob center
            radius (int, optional): Radius of the Gaussian blob. Defaults to 10.
            amplitude (float, optional): Maximum intensity of the blob. Defaults to 1.0.
            shape (tuple, optional): Shape of the output tensor (channels, height, width). Defaults to (3, 480, 720).
            device (torch.device, optional): Device to create the tensor on. Defaults to None.
        
        Returns:
            torch.Tensor: Tensor of shape (channels, height, width) containing the Gaussian blob
        """
        num_channels, height, width = shape
        
        # Create a new tensor filled with zeros
        blob_tensor = torch.zeros(shape, device=device)
        
        # Create coordinate grids
        y_grid, x_grid = torch.meshgrid(
            torch.arange(height, device=device),
            torch.arange(width, device=device),
            indexing='ij'
        )
        
        # Calculate squared distance from (x, y)
        squared_dist = (x_grid - x) ** 2 + (y_grid - y) ** 2
        
        # Create Gaussian blob using the squared distance
        gaussian = amplitude * torch.exp(-squared_dist / (2.0 * radius ** 2))
        
        # Add the Gaussian blob to all channels
        for c in range(num_channels):
            blob_tensor[c] = gaussian
        
        return blob_tensor


    def load_controlnet_signal(self, force, angle, x_pos, y_pos, num_frames=49, num_channels=3, height=480, width=720):

        controlnet_signal = torch.zeros((num_frames, num_channels, height, width)) # (49, 3, 480, 720)

        x_pos_start = x_pos*width
        y_pos_start = (1-y_pos)*height

        DISPLACEMENT_FOR_MAX_FORCE = width / 2
        DISPLACEMENT_FOR_MIN_FORCE = width / 8

        force_percent = (force - self.min_force) / (self.max_force - self.min_force)
        total_displacement = DISPLACEMENT_FOR_MIN_FORCE + (DISPLACEMENT_FOR_MAX_FORCE - DISPLACEMENT_FOR_MIN_FORCE) * force_percent

        x_pos_end = x_pos_start + total_displacement * math.cos(angle * torch.pi / 180.0)
        y_pos_end = y_pos_start - total_displacement * math.sin(angle * torch.pi / 180.0)

        for frame in range(num_frames):

            t = frame / (num_frames-1)
            x_pos_ = x_pos_start * (1-t) + x_pos_end * t # t = 0 --> start; t = 0 --> end
            y_pos_ = y_pos_start * (1-t) + y_pos_end * t # t = 0 --> start; t = 0 --> end

            blob_tensor = self.get_gaussian_blob(x=x_pos_, y=y_pos_, radius=20, amplitude=1.0, shape=(3, 480, 720))

            controlnet_signal[frame] += blob_tensor
        
        return controlnet_signal

class ForcePromptingDataset_WindForce(BaseClass):

    def __init__(self, csv_path, is_validation_dataset=False, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.is_validation_dataset = is_validation_dataset

        if is_validation_dataset:
            self.media_type = "image"
            blob_ext =  "*.png"
        else:
            self.media_type = "video"
            blob_ext = "*.mp4"
            
        file_paths = glob.glob(os.path.join(self.video_root_dir, blob_ext))
        file_names = set([os.path.basename(x) for x in file_paths]) # list of videos or images...
        self.df = pd.read_csv(csv_path)

        # only keep the rows in the csv whose videos we can find
        self.df['checked'] = self.df[self.media_type].map(lambda x, files=file_names: int(x in files))
        self.df = self.df[self.df['checked'] == True]

        self.min_force = float(self.df["wind_speed"].min())
        self.max_force = float(self.df["wind_speed"].max())

        self.length = self.df.shape[0]

    def get_batch(self, idx):

        item = self.df.iloc[idx]
        caption = item['caption']
        file_name = item[self.media_type]
        force = item['wind_speed']
        angle = item['wind_angle']
        file_path = os.path.join(self.video_root_dir, file_name)

        if self.media_type == "image":
            pixel_values = self.load_pixel_values_image(file_path) # (1, 3, 480, 720) of torch.float32 in [-1, 1]
            file_id = file_name.split(".png")[0]
        elif self.media_type == "video":
            pixel_values = self.load_pixel_values_video(file_path) # (49, 3, 480, 720) of torch.float32 in [-1, 1]
            file_id = file_name.split(".mp4")[0]

        controlnet_signal = self.load_controlnet_signal(
            force, angle
        )

        return pixel_values, caption, controlnet_signal, force, angle, file_id

    def __getitem__(self, idx):
        while True:
            try:
                pixel_values, caption, controlnet_signal, force, angle, file_id = self.get_batch(idx)
                # video, caption, controlnet_video = self.get_batch(idx)
                break
            except Exception as e:
                print(e) # this prints 'text' incessantly
                idx = random.randint(0, self.length - 1)
            
        pixel_values = [
            resize_for_crop(x, self.height, self.width) for x in [pixel_values]
        ][0]
        pixel_values = [
            transforms.functional.center_crop(x, (self.height, self.width)) for x in [pixel_values]
        ][0]
        data = {
            'file_id' : file_id,
            'video': pixel_values, 
            'caption': caption, 
            'controlnet_video': controlnet_signal,
            'force': force,
            'angle': angle
        }
        return data

    def load_pixel_values_video(self, video_path):

        video_reader = VideoReader(video_path)
        if random.uniform(0, 1) < 0.5:
            indices = np.array([i for i in range(self.sample_n_frames)], dtype=int)
        else:
            indices = np.array([2*i for i in range(self.sample_n_frames)], dtype=int)

        # Get the selected frames
        np_video = video_reader.get_batch(indices).asnumpy() # (49, 480, 720, 3)
        pixel_values = torch.from_numpy(np_video).permute(0, 3, 1, 2).contiguous() # (49, 3, 480, 720) of uint8 in [0, 255]
        pixel_values = pixel_values / 127.5 - 1 # (49, 3, 480, 720) of torch.float32 in [-1, 1]
        del video_reader

        return pixel_values

    def load_controlnet_signal(self, force, angle, num_frames=49, num_channels=3, height=480, width=720):

        controlnet_signal = torch.zeros((num_frames, num_channels, height, width)) # (49, 3, 480, 720)

        # first channel gets wind_speed
        controlnet_signal[:, 0] = -1 + 2*(force-self.min_force)/(self.max_force-self.min_force)

        # second channel gets cos(wind_angle)
        controlnet_signal[:, 1] = math.cos(angle * torch.pi / 180.0)

        # third channel gets sin(wind_angle)
        controlnet_signal[:, 2] = math.sin(angle * torch.pi / 180.0)
        
        return controlnet_signal