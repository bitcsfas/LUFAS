import math
import numpy as np
import random
import cv2
import torch
import pandas as pd
from PIL import Image
import os
import imgaug.augmenters as iaa
from torch.utils.data import DataLoader
from pathlib import Path
from torchvision import transforms
import bob.io.base
import h5py
import imageio

class RandomErasing(object):
    '''
    Class that performs Random Erasing in Random Erasing Data Augmentation by Zhong et al. 
    -------------------------------------------------------------------------------------
    probability: The probability that the operation will be performed.
    sl: min erasing area
    sh: max erasing area
    r1: min aspect ratio
    mean: erasing value
    -------------------------------------------------------------------------------------
    '''
    def __init__(self, probability = 0.5, sl = 0.01, sh = 0.05, r1 = 0.5, mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1
       
    def __call__(self, sample):
        img, map_x, spoofing_label = sample['image_x'], sample['map_x'],sample['spoofing_label']
        if random.uniform(0, 1) < self.probability:
            attempts = np.random.randint(1, 3)
            for attempt in range(attempts):
                area = img.shape[0] * img.shape[1]
                target_area = random.uniform(self.sl, self.sh) * area
                aspect_ratio = random.uniform(self.r1, 1/self.r1)
                h = int(round(math.sqrt(target_area * aspect_ratio)))
                w = int(round(math.sqrt(target_area / aspect_ratio)))
                if w < img.shape[1] and h < img.shape[0]:
                    x1 = random.randint(0, img.shape[0] - h)
                    y1 = random.randint(0, img.shape[1] - w)
                    img[x1:x1+h, y1:y1+w, 0] = self.mean[0]
                    img[x1:x1+h, y1:y1+w, 1] = self.mean[1]
                    img[x1:x1+h, y1:y1+w, 2] = self.mean[2]
        sample['image_x'] = img
        return sample

class Cutout(object):
    def __init__(self, length=50):
        self.length = length

    def __call__(self, sample):
        img, map_x, spoofing_label = sample['image_x'], sample['map_x'],sample['spoofing_label']
        h, w = img.shape[1], img.shape[2]    # Tensor [1][2],  nparray [0][1]
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)
        length_new = np.random.randint(1, self.length)
        y1 = np.clip(y - length_new // 2, 0, h)
        y2 = np.clip(y + length_new // 2, 0, h)
        x1 = np.clip(x - length_new // 2, 0, w)
        x2 = np.clip(x + length_new // 2, 0, w)
        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        sample['image_x'] = img
        return sample

class Normaliztion(object):
    """
        same as mxnet, normalize into [-1, 1]
        image = (image - 127.5)/128
    """
    def __call__(self, sample):
        image_x, map_x, spoofing_label = sample['image_x'], sample['map_x'],sample['spoofing_label']
        new_image_x = (image_x - 127.5)/128     # [-1,1]
        new_map_x = map_x/255.0                 # [0,1]
        sample['image_x'] = new_image_x
        sample['map_x'] = new_map_x
        return sample

class RandomHorizontalFlip(object):
    """Horizontally flip the given Image randomly with a probability of 0.5."""
    def __call__(self, sample):
        image_x, map_x, spoofing_label = sample['image_x'], sample['map_x'],sample['spoofing_label']
        new_image_x = np.zeros((256, 256, 3))
        new_map_x = np.zeros((32, 32))
        p = random.random()
        if p < 0.5:
            #print('Flip')
            new_image_x = cv2.flip(image_x, 1)
            new_map_x = cv2.flip(map_x, 1)
            sample['image_x'] = new_image_x
            sample['map_x'] = new_map_x
        return sample

class ToTensor(object):
    """
        Convert ndarrays in sample to Tensors.
        process only one batch every time
    """
    def __call__(self, sample):
        image_x, map_x, spoofing_label = sample['image_x'], sample['map_x'],sample['spoofing_label']
        image_raw = sample['image_raw']
        # swap color axis because
        # numpy image: (batch_size) x H x W x C
        # torch image: (batch_size) x C X H X W
        image_x = image_x[:,:,::-1].transpose((2, 0, 1))
        image_x = np.array(image_x)
        map_x = np.array(map_x)
        image_raw = image_raw.transpose((2, 0, 1))
        image_raw = np.array(image_raw)
        spoofing_label_np = np.array([0],dtype=np.int64)
        spoofing_label_np[0] = spoofing_label
        sample['image_x'] = torch.from_numpy(image_x.astype(np.float64)).float()
        sample['map_x'] = torch.from_numpy(map_x.astype(np.float64)).float()
        sample['image_raw'] = torch.from_numpy(image_raw.astype(np.float64)).float()
        sample['spoofing_label'] = torch.from_numpy(spoofing_label_np.astype(np.int64)).long()
        return sample



def crop_face_from_scene(image,face_name_full, scale):
    f = open(face_name_full,'r')
    lines = f.readlines()
    y1,x1,w,h = [float(ele) for ele in lines[:4]]
    f.close()
    y2 = y1+w
    x2 = x1+h
    y_mid = (y1+y2)/2.0
    x_mid = (x1+x2)/2.0
    h_img, w_img  =  image.shape[0], image.shape[1]
    #w_img,h_img = image.size
    w_scale = scale*w
    h_scale = scale*h
    y1 = y_mid-w_scale/2.0
    x1 = x_mid-h_scale/2.0
    y2 = y_mid+w_scale/2.0
    x2 = x_mid+h_scale/2.0
    y1 = max(math.floor(y1),0)
    x1 = max(math.floor(x1),0)
    y2 = min(math.floor(y2),w_img)
    x2 = min(math.floor(x2),h_img)
    #region = image[y1:y2,x1:x2]
    region = image[x1:x2,y1:y2]
    return region

class DataLoaderWMCA(object):
    """data loader for OULU dataset
    """
    def __init__(self, root, path, map_dir = None, protocol=None, transform=None, imgType="gray", imgSize=(64,64), training_state=0, sampling_size=1):
        print("--------------------------WMCA Datset--------------------------")
        self.root           = root
        self.path           = path
        self.map_dir        = map_dir
        self.protocol       = protocol
        self.imgSize        = imgSize
        self.imgType        = imgType
        self.training_state = training_state
        self.sampling_size  = 1 if self.training_state==0 else sampling_size
        if transform==True :
            self.transform = transforms.Compose([RandomErasing(), RandomHorizontalFlip(), ToTensor(), Cutout(), Normaliztion()])
        elif transform==False :
            self.transform = transforms.Compose([ToTensor(), Normaliztion()])
        else :
            raise Exception("transform should be either True or False.")
        path = root+path
        print("Path is "+path)
        print("Protocol is "+self.protocol)
        self.csv_path = root+"/WMCA/wmca-protocols-csv/PROTOCOL-"+self.protocol+".csv"
        print("CSV path is "+self.csv_path)
        self.df = self._read_dataFrame(self.csv_path)
        self.video_count = self.df.shape[0]
        print("{} videos in this protocol".format(self.video_count))
        print("---------------------------------------------------------------")
    
    def _read_dataFrame(self, file_path):
        df = pd.read_csv(file_path)
        tmp = []
        for index,row in df.iterrows():
            if(self.training_state==0):
                if(row[3]=="train"):
                    tmp.append(row)
            if(self.training_state==1):
                if(row[3]=="dev"):
                    tmp.append(row)
            if(self.training_state==2):
                if(row[3]=="eval"):
                    tmp.append(row)
        df = pd.DataFrame(tmp)
        return df
    
    def __len__(self):
        return self.video_count
    
    def _get_cost_time(self, diff) :
        second = diff % 60
        minute = diff % 3600
        minute = minute // 60
        hour = diff % 86400
        hour = hour // 3600
        day = diff // 86400
        format_string="{}d {}h:{}m:{}s"
        return format_string.format(day, hour, minute, second)
    
    def _transform_color(self, img, imgType):
        """function: transform the color space
            args:
                img: PIL.Image
                imgType: string, "gray", "rgb", "hsv", "luv", "lab", "ycrcb", "hls", "private".
            return:
                image in target color space
        """
        if imgType.lower() == "gray" :
            img = img.convert("L")
            img = np.array(img)
            img = img[..., np.newaxis]
        elif imgType.lower() == "rgb" :
            if img.mode != "RGB":
                img = np.array(img)
                img = np.dstack((img, img, img))
            else :
                img = np.array(img)
        elif imgType.lower() == "cmyk" :
            img = img.convert("CMYK")
            img = np.array(img)
        elif imgType.lower() == "hsv" :
            img = np.array(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif imgType.lower() == "luv" :
            img = np.array(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif imgType.lower() == "lab" :
            img = np.array(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        elif imgType.lower() == "ycrcb" :
            img = np.array(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        elif imgType.lower() == "hls" :
            img = np.array(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif imgType.lower() == "hsvycrcb" or imgType.lower() == "ycrcbhsv" :
            img = np.array(img)
            img = np.concatenate((cv2.cvtColor(img, cv2.COLOR_RGB2HSV), cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)), axis=2)
        elif imgType.lower() == "rgbhsv" or imgType.lower() == "hsvrgb" :
            img = np.array(img)
            img = np.concatenate((img, cv2.cvtColor(img, cv2.COLOR_RGB2HSV)), axis=2)
        elif imgType.lower() == "labhsv" or imgType.lower() == "hsvlab" :
            img = np.array(img)
            temp_LAB = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            temp_HSV = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            img = np.concatenate((temp_HSV,temp_LAB), axis=2)
        else:
            raise Exception("wrong image type!")
        return img
    
    def _straighten_face(self, eyes, img) :
        """function: straighten the face image according to the locations of eyes
            paras:
                eyes: [[x1,y1], [x2,y2]], the locations of eyes
                img:  PIL Image
            return:
                rotated image
        """
        theta  = np.arctan(-(eyes[0,1]-eyes[1,1])/(eyes[0,0]-eyes[1,0]))
        temp_img = img.rotate( -theta / np.pi * 180 )
        return temp_img
    
    def _path2label(self, path) :
        """function: get the label from the path"""
        end = path.rfind("/")
        start = path[:end].rfind("_")
        res = path[start+1:end]
        return int(res)
    
    def _getlabel(self, data) :
        if "pseudo_label" not in data.keys() or np.isnan(data["pseudo_label"]) :
            path = data["path"]
            return self._path2label(path)
        else :
            return data["pseudo_label"]
    
    def _process_pipeLine(self, path, imgType, imgSize,augment) :
        """function: process the current path, to get a normalized face and corresponding data
            args:
                path:           path of image
                imgType:        target image color space, "gray", "rgb", "hsv", "luv", "lab", "ycrcb", "hls", "private"
                imgSize:        target image size, (0,0) means no change
                if_alignment:   whether to make the face vertical
                eyes:           position of eyes of each frame, [[left_x,left_y], [right_x,right_y]]
                bbox:           position of bbox of each frame after alignment, (top, right, down, left)
            return:
                img: the processed image
        """
        # read image
        data = h5py.File(path, 'r')
        slice = random.sample(list(data.keys()), 1)
        path=slice[0]+"/array"
        raw=np.array(data.get(path))
        if(raw.shape[0]==3):
            img = cv2.merge([raw[0],raw[1],raw[2]])
            img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
        elif(raw.shape[0]==4):
            img = raw[1]
            img = Image.fromarray(img)
        data.close()
        # resize image
        if img.size != imgSize and imgSize[0] != 0 :
            img = img.resize(imgSize)
        # color space transformation
        if imgType != None :
            img = self._transform_color(img, imgType)
        else :
            img = np.array(img, dtype=np.float32)
        # augment
        if augment == True :
            # data augment from 'imgaug' --> Add (value=(-40,40), per_channel=True), GammaContrast (gamma=(0.5,1.5))
            seq = iaa.Sequential([
                iaa.Add(value=(-40,40), per_channel=True), # Add color 
                iaa.GammaContrast(gamma=(0.5,1.5)) # GammaContrast with a gamma of 0.5 to 1.5
            ])
            img_aug = seq.augment_image(img) 
            return img, img_aug,slice[0]
        else :
            return img

    def __getitem__(self, idx):
        img_list  = []
        map_list  = []
        frame_list = []
        item = self.df.iloc[idx]
        # print(item[0])
        for sampling_id in np.arange(self.sampling_size) :
            img_raw, img ,frame_id  = self._process_pipeLine(self.root+"/WMCA/preprocessed-face-station_RGB/"+item[0]+".hdf5", self.imgType,self.imgSize, augment=True)
            if item[1] == 0:
                label = 1
            else:
                label = 0
            # prepare the ground truth for training or evaluation
            # here we only use binary supervision for convenience, a more powerful supervision, like depth map, could be further considered.
            if label == 1 :
                spoofing_label = 1            # real
                map_x = np.ones((32, 32), dtype=np.float32)
                # directories of depth map files
                if self.map_dir :
                    # print("Get depth")
                    map_x  = self._process_pipeLine(self.root+"/WMCA/preprocessed-face-station_CDIT/"+item[0]+".hdf5", None,(32,32), augment=False)
                    # raise Exception("The process about depth maps is still in building, please switch to map_dir=None mode.")
            else :
                spoofing_label = 0
                map_x = np.zeros((32, 32), dtype=np.float32)    # fake
            # data augmentation
            sample = {"image_raw":img_raw, 'image_x': img, 'map_x': map_x, 'spoofing_label': spoofing_label,'frame_id':frame_id,'path':item[0]}
            sample = self.transform(sample)
            img_list.append( sample['image_x'] )
            map_list.append( sample['map_x'] )
            frame_list.append( sample['frame_id'] )
        
        if self.training_state==0 :
            sample = {"image_raw":sample["image_raw"], 'image_x': img_list[0], 'map_x': map_list[0], 'spoofing_label': spoofing_label}
        else :
            sample = {"image_raw":sample["image_raw"], 'image_x': torch.stack(img_list,axis=0), 'map_x': torch.stack(map_list,axis=0), 'spoofing_label': spoofing_label,'frame_list':frame_list,'path':item[0]}
        return sample

if __name__ == '__main__':
    print("test")