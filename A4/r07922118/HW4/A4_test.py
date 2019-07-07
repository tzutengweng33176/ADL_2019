# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 08:16:32 2018

@author: USER
"""

import torch
import torch.nn
import torch.optim as optim
import torchvision.transforms as Transform
from torchvision.utils import save_image

import numpy as np
import os

import datasets
import ACGAN_WGAN_split
import utils
from argparse import ArgumentParser

hair_mapping =  ['blonde', 'orange', 'brown', 'black', 'blue', 'white']
hair_dict = {
    'blonde' : 0,
    'orange': 1, 
    'brown': 2,
    'black': 3,
    'blue': 4,
    'white': 5
}

eye_mapping = ['brown', 'blue', 'green', 'black']
eye_dict = {
    'brown': 0,
    'blue': 1,
    'green': 2,
    'black': 3
}

face_mapping = ['African', 'Asian', 'Caucasian']
eye_dict = {
    'African': 0,
    'Asian': 1,
    'Caucasian': 2
}

glass_mapping = ['with_glasses', 'without_glasses']
eye_dict = {
    'with_glasses': 0,
    'without_glasses': 1
}


parser = ArgumentParser()
parser.add_argument('-t', '--type', help = 'Type of cartoon generation.', 
                    choices = ['fix_noise', 'fix_hair_eye'], 
                    default = 'fix_noise', type = str)
parser.add_argument('-s', '--sample_dir', help = 'Folder to save the generated samples.',
                    default = './generated', type = str)
parser.add_argument('-d', '--model_dir', help = 'Folder where the trained model is saved',
                    default = './checkpoint', type = str)
parser.add_argument('-l', '--label_dir', help = 'Folder to load the labels',
                     type = str)
args = parser.parse_args()




        
def main():
    f= open(args.label_dir, 'r')
    num_of_samples= f.readline()
    print(f.readline())
    tags_list=[]
    #results=[]
    for line in f:
        #print(line)
        #print(line.split())
        line= line.split()
        #print(line[0])
        tags= list(map(int, line[:]))
#        print(tags)
        #print(len(tags))
        #print(np.asarray(tags))
        tags_list.append(torch.from_numpy(np.asarray(tags)))
#        print(tags_list)
#        input()
    f.close()
    #print(len(tags_list)) #correct
    #input()
    if not os.path.exists(args.sample_dir):
        os.mkdir(args.sample_dir)
    latent_dim = 100
    hair_classes = 6
    eye_classes = 4
    face_classes = 3
    glass_classes = 2
    batch_size = 1

    device = 'cuda'
    G_path = '{}/G_50000.ckpt'.format(args.model_dir)
    #load trained generator here

    G = ACGAN_WGAN_split.Generator(latent_dim = latent_dim, class_dim = hair_classes + eye_classes+face_classes +glass_classes)
    prev_state = torch.load(G_path)
    G.load_state_dict(prev_state['model'])
    G = G.eval()
    G=G.to(device)
    num=0
    for tag in tags_list:
        #randomly generate a noise
        z = torch.randn(latent_dim).unsqueeze(0).to(device)
        #print(z.shape)
        tag= tag.unsqueeze(0).to(device)
        tag= tag.type(torch.float32)
        #print(tag.shape)
        #print(tag)
        #input()
        img= G(z, tag) 
        save_image(utils.denorm(img), '{}/{}.png'.format(args.sample_dir,num))
        num+=1
    print(num)

    
if __name__ == "__main__":
    main()
