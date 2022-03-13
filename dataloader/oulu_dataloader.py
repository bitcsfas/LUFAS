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
    f=open(face_name_full,'r')
    lines=f.readlines()
    y1,x1,w,h=[float(ele) for ele in lines[:4]]
    f.close()
    y2=y1+w
    x2=x1+h
    y_mid=(y1+y2)/2.0
    x_mid=(x1+x2)/2.0
    h_img, w_img = image.shape[0], image.shape[1]
    #w_img,h_img=image.size
    w_scale=scale*w
    h_scale=scale*h
    y1=y_mid-w_scale/2.0
    x1=x_mid-h_scale/2.0
    y2=y_mid+w_scale/2.0
    x2=x_mid+h_scale/2.0
    y1=max(math.floor(y1),0)
    x1=max(math.floor(x1),0)
    y2=min(math.floor(y2),w_img)
    x2=min(math.floor(x2),h_img)
    #region=image[y1:y2,x1:x2]
    region=image[x1:x2,y1:y2]
    return region

class DataLoaderOULU(object):
    """data loader for OULU dataset
    """
    def __init__(self, root, path, map_dir=None, protocol=None, transform=None, imgType="gray", imgSize=(256,256), if_alignment=False, enlarge=0, training_state=0, sampling_size=1, out_sequence=False, **kwargs):
        """function: group the iamges and labels, each label is crossponding to a group containing a set of images
            
            args:
                root:           the root of dataset;
                path:           the path of path list;
                map_dir:        the path of depth map list;
                protocol:       the protocol to be used in this evalutation or training, like "protocol_x" or just x;
                transform:      whether using additional augmentation;
                imgType:        target image color space, "gray", "rgb", "hsv", "luv", "lab", "ycrcb", "hls", "private";
                imgSize:        target image size, (0,0) means no change;
                if_alignment:   whether to make the face vertical;
                enlarge:        enlarge the bbox from each side, top, right, down and left;
                training_state: 0: training, 1: validation, 2: testing;
                sampling_size:  the size of frame sampling for validation or testing;
                out_sequence:   the original output is dict, change it into sequence if out_sequence is True;
                kwargs:         more arguments for different protocols, like leave_out_phone;
        """
        print("--------------------------OULU-NPU Datset--------------------------")
        self.root           = root
        self.csv_path       = path
        self.map_dir        = map_dir
        self.protocol       = protocol
        self.imgSize        = imgSize
        self.imgType        = imgType
        self.if_alignment   = if_alignment
        self.enlarge        = enlarge
        self.training_state = training_state
        self.sampling_size  = 1 if self.training_state==0 else sampling_size
        self.out_sequence   = out_sequence
        self.kwargs         = kwargs
        
        if transform==True :
            self.transform = transforms.Compose([RandomErasing(), RandomHorizontalFlip(), ToTensor(), Cutout(), Normaliztion()])
        elif transform==False :
            self.transform = transforms.Compose([ToTensor(), Normaliztion()])
        else :
            raise Exception("transform should be either True or False.")
        self._safe_check()
        # read dataset
        self._df, self.groups, self.keys = self._read_dataset_paths(self.csv_path)
        self._df, self.groups, self.keys = self._prorocal_split(self.protocol, self._df, self.groups, self.keys, self.training_state, self.kwargs)
        
        self._dataset_size = len(self._df)
        print("there are {} paths and {} groups with {}.".format(self._dataset_size, len(self.keys), self.protocol))
        print("imgSize:{}, imgType:{}, if_alignment:{}, self.enlarge:{}".format(self.imgSize, self.imgType, self.if_alignment, self.enlarge))
        print("training_state:{}, sampling_size:{}".format(self.training_state, self.sampling_size))
        print("kwargs: {}".format(self.kwargs))
    
    def _safe_check(self):
        if self.protocol=="protocol_3" or self.protocol==3 or self.protocol=="protocol_4" or self.protocol==4 :
            assert self.kwargs["leave_out_phone"] in [1,2,3,4,5,6]
        assert self.training_state in [0,1,2,3]
    
    def _read_dataset_paths(self, csv_path):
        # read file
        df           = self._read_dataFrame(csv_path)
        df           = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        df["prefix"] = df[["path"]].applymap(lambda ele: ele[:ele.rfind("/")] )
        groups       = df.groupby("prefix")
        keys         = list(groups.groups.keys())
        return df, groups, keys
    
    def _read_dataFrame(self, file_path):
        df = pd.read_csv(file_path)
        return df
    
    def _prorocal_split(self, protocol, df, groups, keys, training_state, kwargs):
        # processing the dataset according to the official description
        if protocol is None or protocol=="None" or protocol=="self" :
            pass
        elif protocol=="protocol_1" or protocol==1 :
            if self.training_state==2 :
                df = df[ df["session_type"].isin([3]) ]
            else :
                df = df[ df["session_type"].isin([1,2]) ]
            
        elif protocol=="protocol_2" or protocol==2 :
            if self.training_state==2 :
                df = df[ df["access_type"].isin([1,3,5]) ]
            else :
                df = df[ df["access_type"].isin([1,2,4]) ]
            
        elif protocol=="protocol_3" or protocol==3 :
            if self.training_state==2 :
                df = df[ df["phone_type"].isin( [kwargs["leave_out_phone"]] ) ]
            else :
                df = df[ df["phone_type"].isin( np.arange(1,7)[np.arange(1,7)!=kwargs["leave_out_phone"]] ) ]
            
        elif protocol=="protocol_4" or protocol==4 :
            if self.training_state==2 :
                df = df[ df["session_type"].isin([3]) &
                         df["access_type"].isin([1,3,5]) & 
                         df["phone_type"].isin( [kwargs["leave_out_phone"]] ) ]
            else :
                df = df[ df["session_type"].isin([1,2]) &
                         df["access_type"].isin([1,2,4]) &
                         df["phone_type"].isin( np.arange(1,7)[np.arange(1,7)!=kwargs["leave_out_phone"]] ) ]
            
        else :
            raise Exception("No such protocol: {}".format(protocol))
        
        if self.kwargs.get("pseudo_csv") is not None :
            pseudo_df = self._read_dataFrame(self.kwargs.get("pseudo_csv"))
            df = pd.concat([df, pseudo_df], ignore_index=True)
            print("df: {}, pseudo_df: {}, all: {}".format(len(list(df.groupby("prefix").groups.keys()))-len(list(pseudo_df.groupby("prefix").groups.keys())), len(list(pseudo_df.groupby("prefix").groups.keys())), len(list(df.groupby("prefix").groups.keys()))))
        
        groups = df.groupby("prefix")
        keys   = list(groups.groups.keys())
        return df, groups, keys
    
    def __len__(self):
        return len(self.keys)
    
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
    
    def _crop_face(self, img, bbox, enlarge=0):
        """function: get the face area
            paras:
                img:     PIL Image
                bbox:    [top, right, down, left]
                enlarge: enlarge each side;
            return:
                face image
        """
        H,W = img.size
        top, right, down, left = bbox
        if enlarge==0 :
            pass
        elif enlarge>0 and enlarge<=1 :
            radius  = max(right-left, down-top)
            enlarge = enlarge*radius
            top     = top-enlarge if top-enlarge>0 else 0
            right   = right+enlarge if right+enlarge<W else W
            down    = down+enlarge if down+enlarge<H else H
            left    = left-enlarge if left-enlarge>0 else 0
        elif enlarge>1 :
            top     = top-enlarge if top-enlarge>0 else 0
            right   = right+enlarge if right+enlarge<W else W
            down    = down+enlarge if down+enlarge<H else H
            left    = left-enlarge if left-enlarge>0 else 0
        elif enlarge<0 :
            radius  = max(right-left, down-top)
            if self.training_state == 0 :
                enlarge = np.random.randint(1,3)/10.0 * radius
            else :
                enlarge = 0.2 * radius
            top     = top-enlarge if top-enlarge>0 else 0
            right   = right+enlarge if right+enlarge<W else W
            down    = down+enlarge if down+enlarge<H else H
            left    = left-enlarge if left-enlarge>0 else 0
        
        if right-left > down-top :
            dis = ( right-left - (down-top) ) / 2
            down += dis
            top -= dis
        elif right-left < down-top :
            dis = ( down-top - (right-left) ) / 2
            right += dis
            left -= dis
        face = img.crop( (left, top, right, down) )
        return face
    
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
    
    def _path2types(self, path) :
        """function: get the types of phone, session, user and access from the path"""
        end = path.rfind("/")
        start = path[:end].rfind("_")
        access = int(path[start+1:end])
        end = start
        start = path[:end].rfind("_")
        user = int(path[start+1:end])
        end = start
        start = path[:end].rfind("_")
        session = int(path[start+1:end])
        end = start
        start = path[:end].rfind("/")
        phone = int(path[start+1:end])
        return [phone, session, user, access]

    def _process_pipeLine(self, path, imgSize, imgType, if_alignment=False, eyes=None, bbox=None, enlarge=0, augment=True) :
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
        img = Image.open(path)
        # align the face
        if if_alignment is True :
            if eyes is None or bbox is None :
                raise Exception("Please give the eyes and bbox")
            # check the data in case of the error alignment
            if (eyes==0).sum()>=2 or (bbox==0).sum()>=2 :
                return None
            img = self._straighten_face(eyes, img)
            img = self._crop_face(img, bbox, enlarge=enlarge)
        else :
            if bbox is None :
                raise Exception("Please give the bbox")
            img = self._crop_face(img, bbox, enlarge=enlarge)
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
            return img, img_aug
        else :
            return img
    
    def __getitem__(self, idx):
        img_list  = []
        map_list  = []
        index_list = []
        for sampling_id in np.arange(self.sampling_size) :
            cnt = 0
            while True :
                chs  = np.random.randint( len( self.groups.get_group(self.keys[idx]) ) )
                data = self.groups.get_group(self.keys[idx]).iloc[chs]
                path = os.path.join(self.root, data["path"])
                if self.if_alignment :
                    eyes_pos = data[['eye_left_x', 'eye_left_y', 'eye_right_x', 'eye_right_y']].values.reshape((2,2))
                    bbox_pos = data[['loc_rotation_top', 'loc_rotation_right', 'loc_rotation_down', 'loc_rotation_left']].values
                else :
                    eyes_pos = None
                    try :
                        bbox_pos = np.array(list(map(float, data['loca'][1:-1].split(","))))
                    except Exception as err :
                        print("error: {}".format(err))
                        print("path: {}".format(path))
                        print("loca: {}".format(data['loca']))
                # once the bbox is not corrupted, pick the current image as a sampling result
                if bbox_pos.sum() > 10 :
                    break
                cnt += 1
                # if the bbox is still corrupted after several iterations, the image is most likely to be attack data
                if cnt >= 30 :
                    raise Exception("there is something wrong with {}".format(self.keys[idx]))
            img_raw, img   = self._process_pipeLine(path, self.imgSize, self.imgType, if_alignment=self.if_alignment, eyes=eyes_pos, bbox=bbox_pos, enlarge=self.enlarge, augment=True)
            label          = self._getlabel(data)
            types          = self._path2types(path)
            key            = self.keys[idx]
            # prepare the ground truth for training or evaluation
            # here we only use binary supervision for convenience, a more powerful supervision, like depth map, could be further considered.
            if label == 1 :
                spoofing_label = 1            # real
                map_x = np.ones((32, 32), dtype=np.float32)
                # directories of depth map files
                if self.map_dir is not None :
                    map_path = path.replace(".jpg", "_depth.jpg")
                    depth_file = Path(map_path)
                    if depth_file.exists():
                        map_x    = self._process_pipeLine(map_path, (32,32), None, if_alignment=self.if_alignment, eyes=eyes_pos, bbox=bbox_pos, enlarge=self.enlarge, augment=False)
                    # raise Exception("The process about depth maps is still in building, please switch to map_dir=None mode.")
            else :
                spoofing_label = 0
                map_x = np.zeros((32, 32), dtype=np.float32)    # fake
            
            # data augmentation
            sample = {"image_raw":img_raw, 'image_x': img, 'map_x': map_x, 'spoofing_label': spoofing_label, "types": types, "key": key}
            sample = self.transform(sample)
            img_list.append( sample['image_x'] )
            map_list.append( sample['map_x'] )
            index_list.append( (idx,chs) )
        
        if self.training_state==0 :
            sample = {"image_raw":sample["image_raw"], 'image_x': img_list[0], 'map_x': map_list[0], 'spoofing_label': spoofing_label, "types": types, "key": key}
        elif self.training_state==3 :
            sample = {"image_raw":sample["image_raw"], 'image_x': torch.stack(img_list,axis=0), 'map_x': torch.stack(map_list,axis=0), 'spoofing_label': spoofing_label, "types": types, "key": key, "index_list": np.array(index_list)}
        else :
            sample = {"image_raw":sample["image_raw"], 'image_x': torch.stack(img_list,axis=0), 'map_x': torch.stack(map_list,axis=0), 'spoofing_label': spoofing_label, "types": types, "key": key}
        if self.out_sequence :
            return sample['image_raw'], sample['image_x'], sample['map_x'], sample['spoofing_label'], sample['types'], sample['key']
        else :
            return sample