import torch
import torch.nn
import torch.optim as optim
import torchvision.transforms as Transform
from torchvision.utils import save_image

import numpy as np
import os
from argparse import ArgumentParser

from datasets import Anime, Shuffler
from ACGAN import Generator, Discriminator
from utils import save_model, denorm, plot_loss, plot_classifier_loss, show_process
from utils import generation_by_attributes, get_random_label

parser = ArgumentParser()
parser.add_argument('-d', '--device', help = 'Device to train the model on', 
                    default = 'cuda', choices = ['cuda', 'cpu'], type = str)
parser.add_argument('-i', '--iterations', help = 'Number of iterations to train ACGAN', 
                    default = 300000, type = int)
parser.add_argument('-b', '--batch_size', help = 'Training batch size',
                    default = 16, type = int)
parser.add_argument('-t', '--train_dir', help = 'Training data directory', 
                    default = 'selected_cartoonset100k', type = str)
parser.add_argument('-s', '--sample_dir', help = 'Directory to store generated images', 
                    default = 'samples', type = str)
parser.add_argument('-c', '--checkpoint_dir', help = 'Directory to save model checkpoints', 
                    default = 'checkpoints', type = str)
parser.add_argument('--sample', help = 'Sample every _ steps', 
                    default = 5000, type = int)
parser.add_argument('--check', help = 'Save model every _ steps', 
                    default = 10000, type = int)
parser.add_argument('--lr', help = 'Learning rate of ACGAN. Default: 0.0002', 
                    default = 0.0002, type = float)
parser.add_argument('--beta', help = 'Momentum term in Adam optimizer. Default: 0.5', 
                    default = 0.5, type = float)
parser.add_argument('--classification_weight', help = 'Classification loss weight. Default: 1',
                    default = 1, type = float)
args = parser.parse_args()

if args.device == 'cuda' and not torch.cuda.is_available():
    print("Your device currenly doesn't support CUDA.")
    exit()
print('Using device: {}'.format(args.device))

def main():
    batch_size = args.batch_size
    iterations =  args.iterations
    device = args.device
    
    #hair_classes, eye_classes = 12, 10
    hair_classes, eye_classes, face_classes, glass_classes= 6, 4, 3, 2
    num_classes = hair_classes + eye_classes + face_classes + glass_classes
    latent_dim = 100
    
    config = 'ACGAN-batch_size-[{}]-steps-[{}]'.format(batch_size, iterations)
    print('Configuration: {}'.format(config))
    
    
    root_dir = './{}/images'.format(args.train_dir)
    tags_file = './{}/image_to_tag.pickle'.format(args.train_dir)
    #you need to make a pickle file
    #Each training image is tagged with exactly 4 labels: one from hair tags, 1 from eye tags, 1 from face tags, and 1 from glass class. 
    #In tags.pickle, each image index is associated with a 15-dimensional tensor (which is a multi-label one-hot encoding).
    #you can generate a image_file_name-to-index file and a index-to-tag file.
    #hair_prior = np.load('../{}/hair_prob.npy'.format(args.train_dir)) -->this is optional 
    #eye_prior = np.load('../{}/eye_prob.npy'.format(args.train_dir))
    #maybe in each class we can sum the number of each tag and calculate their appearing times to get the probability distribution
    #But I think we should deal with this issue later
    #see https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.random.choice.html 
    random_sample_dir = './{}/{}/random_generation'.format(args.sample_dir, config)
    fixed_attribute_dir = './{}/{}/fixed_attributes'.format(args.sample_dir, config)
    checkpoint_dir = './{}/{}'.format(args.checkpoint_dir, config)
    
    if not os.path.exists(random_sample_dir):
    	os.makedirs(random_sample_dir)
    if not os.path.exists(fixed_attribute_dir):
    	os.makedirs(fixed_attribute_dir)
    if not os.path.exists(checkpoint_dir):
    	os.makedirs(checkpoint_dir)
        
    ########## Start Training ##########

    transform = Transform.Compose([Transform.ToTensor(),
                                   Transform.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    dataset = Anime(root_dir = root_dir, tags_file = tags_file, transform = transform)
    shuffler = Shuffler(dataset = dataset, batch_size = args.batch_size)
    
    G = Generator(latent_dim = latent_dim, class_dim = num_classes).to(device)
    D = Discriminator(num_classes = num_classes).to(device)

    G_optim = optim.Adam(G.parameters(), betas = [args.beta, 0.999], lr = args.lr)
    D_optim = optim.Adam(D.parameters(), betas = [args.beta, 0.999], lr = args.lr)
    
    d_log, g_log, classifier_log = [], [], []
    criterion = torch.nn.BCELoss()
#when calculating the classification loss, I think we should use nn.NLLLoss here
#for more detail, see https://github.com/gitlimlab/ACGAN-PyTorch/blob/master/main.py
    for step_i in range(1, iterations + 1):

        real_label = torch.ones(batch_size).to(device)
        fake_label = torch.zeros(batch_size).to(device)
        
        # Train discriminator
        #update D network first
        D.zero_grad() #location of zero_grad is strange
        real_img, hair_tags, eye_tags, face_tags, glass_tags = shuffler.get_batch()
        real_img, hair_tags, eye_tags, face_tags, glass_tags = real_img.to(device), hair_tags.to(device), eye_tags.to(device), face_tags.to(device), glass_tags.to(device)
        real_tag = torch.cat((hair_tags, eye_tags, face_tags, glass_tags), dim = 1)
        real_tag= real_tag.type(torch.float32) #cast the type from int64 to float32
        #print(real_tag.shape) #torch.Size([128, 15])
        #input()

        z = torch.randn(batch_size, latent_dim).to(device) #noise
        fake_tag = get_random_label(batch_size = batch_size,  #noise label
                                    hair_classes = hair_classes, hair_prior = None,
                                    eye_classes = eye_classes, eye_prior = None, 
                                    face_classes= face_classes, face_prior= None,
                                    glass_classes= glass_classes, glass_prior=None).to(device)
        #print(fake_tag[0]) 
        #print(fake_tag.shape) #torch.Size([128, 15])
        #input()
        fake_img = G(z, fake_tag).to(device) 
        #print(fake_img)
        #print(fake_img.shape) #torch.Size([128, 3, 64, 64]) --> WRONG I need (batch_size, 3, 128, 128) -->SOLVED
        #input()
        #print(real_img.shape ) #torch.Size([128, 3, 128, 128])
        #input()
        real_score, real_predict = D(real_img)
        fake_score, fake_predict = D(fake_img.detach()) #you need detach() here
        #print(real_score.shape) #torch.Size([128])
        #print(real_predict.shape) #torch.Size([128, 15])
        #input()
        #print(real_label) #all 1's
        #print(fake_label) #all 0's
        #print(real_label.shape) #torch.Size([128])
        #print(fake_label.shape) #torch.Size([128])
        #input()
        real_discrim_loss = criterion(real_score, real_label) #-log(P(S=real|X_real))
        fake_discrim_loss = criterion(fake_score, fake_label) #-log(1-P(S=fake|x_fake))
        #print(real_discrim_loss) #tensor(0.6941, grad_fn=<BinaryCrossEntropyBackward>)
        #print(real_discrim_loss.shape) #torch.Size([])
        #print(fake_discrim_loss) #tensor(0.6695, grad_fn=<BinaryCrossEntropyBackward>)
        #print(fake_discrim_loss.shape) #torch.Size([])
        #input()
        #print(real_predict.dtype)
        #print(real_tag.dtype)
        #print(real_tag)
        #print(real_tag.shape) #torch.Size([128, 15])
        #input()
        real_classifier_loss = criterion(real_predict, real_tag) #--> got some problems here!!! #Lc in the original paper, but 
        #The classification loss for the synthesized image was omitted, since we believe the fake image would confuse the discriminator.
        #print(real_classifier_loss) #tensor(0.6968, grad_fn=<BinaryCrossEntropyBackward>)
        #what the fuck?!
        fake_classifier_loss= criterion(fake_predict, fake_tag)
        discrim_loss = (real_discrim_loss + fake_discrim_loss) * 0.5 #Ls in the original paper; or adversarial loss
        classifier_loss = real_classifier_loss * args.classification_weight+ fake_classifier_loss #or auxiliary_loss
        
        classifier_log.append(classifier_loss.item())
            
        D_loss = discrim_loss + classifier_loss
        #D_optim.zero_grad() #location of zero_grad is strange
        D_loss.backward()
        D_optim.step()

        # Train generator
        #update G network
        G.zero_grad()
        #z = torch.randn(batch_size, latent_dim).to(device)
        #fake_tag = get_random_label(batch_size = batch_size, 
        #                            hair_classes = hair_classes, hair_prior = None,
        #                            eye_classes = eye_classes, eye_prior = None,
        #                            face_classes= face_classes, face_prior= None,
        #                            glass_classes= glass_classes, glass_prior=None).to(device)
                                    
        #fake_img = G(z, fake_tag).to(device) #you don't have to generate two fake images
        
        
        fake_score, fake_predict = D(fake_img)
        
        discrim_loss_G = criterion(fake_score, real_label) #discriminate whether the generated image is real or fake
        #-log(log(P(S=fake|X_fake)))
        classifier_loss_G = criterion(fake_predict, fake_tag) #discriminate the class of the fake image
        #-log(P(C=c|X_fake))
        #here the loss function has been modified(both G and D), different from the original ACGAN paper

        G_loss = classifier_loss_G + discrim_loss_G
        #G_optim.zero_grad()
        G_loss.backward()
        G_optim.step()
            
        ########## Updating logs ##########
        d_log.append(D_loss.item())
        g_log.append(G_loss.item())
        show_process(total_steps = iterations, step_i = step_i,
        			 g_log = g_log, d_log = d_log, classifier_log = classifier_log)

        ########## Checkpointing ##########

        if step_i == 1:
            save_image(denorm(real_img[:16,:,:,:]), os.path.join(random_sample_dir, 'real.png'), nrow=4)
            #https://pytorch.org/docs/stable/torchvision/utils.html?highlight=save_image#torchvision.utils.save_image
        if step_i % args.sample == 0:
            save_image(denorm(fake_img[:16,:,:,:]), os.path.join(random_sample_dir, 'fake_step_{}.png'.format(step_i)), nrow=4)
            
        if step_i % args.check == 0:
            save_model(model = G, optimizer = G_optim, step = step_i, log = tuple(g_log), 
                       file_path = os.path.join(checkpoint_dir, 'G_{}.ckpt'.format(step_i)))
            save_model(model = D, optimizer = D_optim, step = step_i, log = tuple(d_log), 
                       file_path = os.path.join(checkpoint_dir, 'D_{}.ckpt'.format(step_i)))

            plot_loss(g_log = g_log, d_log = d_log, file_path = os.path.join(checkpoint_dir, 'loss.png'))
            plot_classifier_loss(log = classifier_log, file_path = os.path.join(checkpoint_dir, 'classifier loss.png'))

            generation_by_attributes(model = G, device = args.device,batch_size=batch_size ,step = step_i, latent_dim = latent_dim, 
                                     hair_classes = hair_classes, eye_classes = eye_classes, face_classes= face_classes, glass_classes=glass_classes,
                                     sample_dir = fixed_attribute_dir) # -->fix this later
    
if __name__ == '__main__':
    main()
            

        

    
    
    
