import os
import cv2
import pickle
import numpy as np
import torch
        
class Anime:
    """ Dataset that loads images and image tags from given folders.

    Attributes:
        root_dir: folder containing training images
        tags_file: a dictionary object that contains class tags of images.
        transform: torch.Transform() object to perform image transformations.
        img_files: a list of image file names in root_dir
        dataset_len: number of training images.
    """

    def __init__(self, root_dir, tags_file, transform):
        with open(tags_file, 'rb') as file:
            self.tags_file = pickle.load(file) 
        self.root_dir = root_dir
        self.transform = transform
        self.img_files = os.listdir(self.root_dir)
        self.dataset_len = len(self.img_files)
        #print(len(self.img_files))
        self.id_to_image={}
        for i, k in enumerate(self.tags_file):
            self.id_to_image[i]=k
        
       
        #print(self.id_to_image)
        #input()
        self.image_to_id={v:k for k, v in self.id_to_image.items()}
        #print(self.image_to_id)
        #print(self.tags_file)
        #input()
    def length(self):
        return self.dataset_len
    
    def get_item(self, idx):
        """ Return '[idx].jpg' and its tags. """

        #print(self.id_to_image[idx])
        #input()
        #print(self.tags_file[self.id_to_image[idx]])
        #input()
        label = self.tags_file[self.id_to_image[idx]]
        #print(label)
        hair_tag=label[0:6] # 6
        eye_tag=label[6:10] #4
        face_tag= label[10:13] #3
        glass_tag =label[13:15] #2
        #print(hair_tag)
        #print(eye_tag)
        #print(face_tag)
        #print(glass_tag)
        #input()
        #print(self.root_dir)
        img_path = os.path.join(self.root_dir, str(self.id_to_image[idx]))
        #print(img_path)
        #input()
        img = cv2.imread(img_path)
        #print(img)
        #print(img.shape) #(128, 128, 3)
        img = img[:, :, (2, 1, 0)]  # Swap B,R channel of np.array loaded with cv2
        #print(img.shape)     #(128, 128, 3)      						 # (BGR -> RGB)
        if self.transform:
            img = self.transform(img) #transform here--> in train.py
    #transform = Transform.Compose([Transform.ToTensor(),
    #                               Transform.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    #transform the image to tensor and normalize
        #print(img)
        #print(img.shape) #torch.Size([3, 128, 128])
        #input()
        return img, hair_tag, eye_tag, face_tag, glass_tag

class Shuffler:
    """ Class that supports andom sampling of training data.

    Attributes:
        dataset: an Anime dataset object.
        batch_size: size of each random sample.
        dataset_len: size of dataset.
    
    """
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.dataset_len = self.dataset.length()
    
    def get_batch(self):
        """ Returns a batch of randomly sampled images and its tags. 

        Args:
            None.

        Returns:
            Tuple of tensors: img_batch, hair_tags, eye_tags
            img_batch: tensor of shape N * 3 * 64 * 64
            hair_tags: tensor of shape N * hair_classes
            eye_tags: tensor of shape N * eye_classes
        """

        indices = np.random.choice(self.dataset_len, self.batch_size)  # Sample non-repeated indices
        img_batch, hair_tags, eye_tags, face_tags, glass_tags = [], [], [], [], []
        for i in indices:
            img, hair_tag, eye_tag, face_tag, glass_tag = self.dataset.get_item(i)
            img_batch.append(img.unsqueeze(0))
            #print(img.shape)
            hair_tags.append(hair_tag.unsqueeze(0))
            #print(hair_tag.shape)
            #print(hair_tag.unsqueeze(0).shape) #torch.Size([1, 6])
            #input()
            eye_tags.append(eye_tag.unsqueeze(0)) #unsqueeze dim-0 for later concatenation
            face_tags.append(face_tag.unsqueeze(0))
            glass_tags.append(glass_tag.unsqueeze(0))
        img_batch = torch.cat(img_batch, 0)
        #print(img_batch)
        #print(img_batch.shape) #torch.Size([batch_size, 3, 128, 128])
        hair_tags = torch.cat(hair_tags, 0)
        #print(hair_tags)
        #print(hair_tags.shape) #torch.Size([128, 6])
        eye_tags = torch.cat(eye_tags, 0)
        #print(eye_tags.shape) #torch.Size([128, 4])
        face_tags= torch.cat(face_tags, 0)
        #print(face_tags.shape) #torch.Size([128, 3])
        glass_tags= torch.cat(glass_tags, 0)
        #print(glass_tags.shape) #torch.Size([128, 2])
        #input()
        return img_batch, hair_tags, eye_tags, face_tags, glass_tags
    
