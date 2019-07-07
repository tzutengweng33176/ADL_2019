import torch
import torch.nn
import torch.optim as optim
import torch.autograd as autograd
import torchvision.transforms as Transform
from torchvision.utils import save_image

import numpy as np
import os
from argparse import ArgumentParser

from datasets import Anime, Shuffler
from ACGAN_WGAN_split import Generator, Discriminator
from utils import save_model, denorm, plot_loss, plot_classifier_loss, show_process
from utils import generation_by_attributes, get_random_label

parser = ArgumentParser()
parser.add_argument('-d', '--device', help = 'Device to train the model on', 
                    default = 'cuda', choices = ['cuda', 'cpu'], type = str)
parser.add_argument('-i', '--iterations', help = 'Number of iterations to train ACGAN', 
                    default = 50000, type = int)
parser.add_argument('-b', '--batch_size', help = 'Training batch size',
                    default = 64, type = int)
parser.add_argument('-t', '--train_dir', help = 'Training data directory', 
                    default = 'selected_cartoonset100k', type = str)
parser.add_argument('-s', '--sample_dir', help = 'Directory to store generated images', 
                    default = 'samples', type = str)
parser.add_argument('-c', '--checkpoint_dir', help = 'Directory to save model checkpoints', 
                    default = 'checkpoints', type = str)
parser.add_argument('--sample', help = 'Sample every _ steps', 
                    default = 500, type = int)
parser.add_argument('--check', help = 'Save model every _ steps', 
                    default = 1000, type = int)
parser.add_argument('--lr', help = 'Learning rate of ACGAN. Default: 0.0002', 
                    default = 0.0001, type = float)
parser.add_argument('--beta', help = 'Momentum term in Adam optimizer. Default: 0.5', 
                    default = 0.5, type = float)
parser.add_argument('--classification_weight', help = 'Classification loss weight. Default: 1',
                    default = 1, type = float)
args = parser.parse_args()

if args.device == 'cuda' and not torch.cuda.is_available():
    print("Your device currenly doesn't support CUDA.")
    exit()
print('Using device: {}'.format(args.device))

def calculate_gradient_penalty(real_images, fake_images, batch_size, D, device):
        eta = torch.FloatTensor(batch_size,1,1,1).uniform_(0,1)
        #real_image.size = batch_size, 3, 128, 128
        eta = eta.expand(batch_size, real_images.size(1), real_images.size(2), real_images.size(3))
        #print(eta)
        #print(eta.shape) #torch.Size([64, 3, 128, 128])
        #if self.cuda:
        eta = eta.to(device)
        #else:
        #    eta = eta

        interpolated = eta * real_images + ((1 - eta) * fake_images)

        #if self.cuda:
        interpolated = interpolated.to(device)
       # else:
       #     interpolated = interpolated

        # define it to calculate gradient
        interpolated.requires_grad= True
        #print(interpolated.requires_grad)
        #input()
        #interpolated = Variable(interpolated, requires_grad=True)

        # calculate probability of interpolated examples
        prob_interpolated, _, _ ,_ ,_ = D(interpolated)

        # calculate gradients of probabilities with respect to examples
        gradients = autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones(
                                   prob_interpolated.size()).to(device), 
                               create_graph=True, retain_graph=True)[0]
        #Computes and returns the sum of gradients of outputs w.r.t. the inputs.
        #print(gradients)
        #print(gradients.shape) #torch.Size([64, 3, 128, 128])
        gradients= gradients.view(gradients.size(0), -1)
        #input()
        #LAMBDA= 10
        grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10
        return grad_penalty


def main():
    batch_size = args.batch_size
    iterations =  args.iterations
    device = args.device
    
    #hair_classes, eye_classes = 12, 10
    hair_classes, eye_classes, face_classes, glass_classes= 6, 4, 3, 2
    num_classes = hair_classes + eye_classes + face_classes + glass_classes
    latent_dim = 100
    smooth= 0.7  #label smoothing--> see https://www.inference.vc/instance-noise-a-trick-for-stabilising-gan-training/  
    config = 'ACGAN_split_soft_WGANGP_20190526-batch_size-[{}]-steps-[{}]'.format(batch_size, iterations)
    print('Configuration: {}'.format(config))
    
    
    root_dir = './{}/images'.format(args.train_dir)
    tags_file = './{}/image_to_tag.pickle'.format(args.train_dir)
    #you need to make a pickle file
    #Each training image is tagged with exactly 4 labels: one from hair tags, 1 from eye tags, 1 from face tags, and 1 from glass class. 
    #In tags.pickle, each image index is associated with a 15-dimensional tensor (which is a multi-label one-hot encoding).
    #you can generate a image_file_name-to-index file and a index-to-tag file.
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
    D = Discriminator(hair_classes = hair_classes, eye_classes= eye_classes, face_classes= face_classes, glass_classes= glass_classes).to(device)
    #print(G)
    #print(D)
    #input()
    #WGAN with gradient clipping uses RMSprop instead of ADAM
    G_optim = optim.Adam(G.parameters(),betas= (args.beta, 0.999) ,lr = args.lr)
    D_optim = optim.Adam(D.parameters(),betas=(args.beta, 0.999) ,lr = args.lr)
    
    d_log, g_log, classifier_log = [], [], []
    criterion = torch.nn.NLLLoss() #when using WGAN, you still need BCELoss to calculate the classifier loss
#when calculating the classification loss, I think we should use nn.NLLLoss here--->WRONG!!!!
#for more detail, see https://github.com/gitlimlab/ACGAN-PyTorch/blob/master/main.py
    one= torch.FloatTensor([1]).to(device)
    mone= (one*-1).to(device)
    #print(one) #tensor([1.], device='cuda:0')
    #print(mone) #tensor([-1.], device='cuda:0')
    #input()
    for step_i in range(1, iterations + 1):

        real_label = torch.ones(batch_size).to(device)
        fake_label = torch.zeros(batch_size).to(device)
        soft_label= torch.Tensor(batch_size).uniform_(smooth, 1).to(device)
        fake_soft_label= torch.Tensor(batch_size).uniform_(0, 0.3).to(device) 
        #if you don't use smooth label, generator will crash
        for p in D.parameters():
            p.requires_grad= True
        # Train discriminator
        #Training discriminator more iterations than generator WGAN
        #for more details see https://github.com/Zeleni9/pytorch-wgan/blob/master/models/wgan_clipping.py
        for d_iter in range(5):
        #update D network first
            D.zero_grad() #location of zero_grad is strange
            
            
            real_img, hair_tags, eye_tags, face_tags, glass_tags = shuffler.get_batch()
            real_img, hair_tags, eye_tags, face_tags, glass_tags = real_img.to(device), hair_tags.to(device), eye_tags.to(device), face_tags.to(device), glass_tags.to(device)
            #hair_tags, eye_tags, face_tags, glass_tags =hair_tags.type(torch.float32), eye_tags.type(torch.float32), face_tags.type(torch.float32), glass_tags.type(torch.float32)
            #print(hair_tags)
            #print(torch.max(hair_tags, 1))
            #print(torch.max(hair_tags, 1)[1])
            #print(glass_tags)
            z_d = torch.randn(batch_size, latent_dim).to(device) #noise
            with torch.no_grad():
                z= z_d ## totally freeze G, training D
                #see https://github.com/jalola/improved-wgan-pytorch/blob/master/congan_train.py
            fake_tag = get_random_label(batch_size = batch_size,  #noise label
                                    hair_classes = hair_classes, hair_prior = None,
                                    eye_classes = eye_classes, eye_prior = None, 
                                    face_classes= face_classes, face_prior= None,
                                    glass_classes= glass_classes, glass_prior=None).to(device)
            fake_hair_tags_d= fake_tag[: , 0:6]
            fake_eye_tags_d=fake_tag[:, 6:10]
            fake_face_tags_d= fake_tag[:, 10:13]
            fake_glass_tags_d= fake_tag[:, 13:15]
            #print(fake_hair_tags)

            real_img.requires_grad= True
            real_score, real_hair_predict, real_eye_predict, real_face_predict, real_glass_predict = D(real_img)
            real_discrim_loss = real_score.mean(0) #-log(P(S=real|X_real))
            
            #real_hair_classifier_loss= criterion(real_hair_predict, hair_tags)
            #real_eye_classifier_loss= criterion(real_eye_predict, eye_tags)
            #real_face_classifier_loss= criterion(real_face_predict, face_tags)
            #real_glass_classifier_loss= criterion(real_glass_predict, glass_tags)
            real_hair_classifier_loss= criterion(real_hair_predict, torch.max(hair_tags, 1)[1]).mean()
            real_eye_classifier_loss= criterion(real_eye_predict, torch.max(eye_tags,1 )[1]).mean()
            real_face_classifier_loss= criterion(real_face_predict, torch.max(face_tags, 1)[1]).mean()
            real_glass_classifier_loss= criterion(real_glass_predict, torch.max(glass_tags, 1)[1]).mean()
            real_classifier_loss = real_hair_classifier_loss +real_eye_classifier_loss+real_face_classifier_loss+ real_glass_classifier_loss
            #print(real_classifier_loss) #tensor(2.3835, device='cuda:0', grad_fn=<AddBackward0>)
            #print(real_classifier_loss.mean()) #tensor(2.3835, device='cuda:0', grad_fn=<MeanBackward1>)
            #input()
            errD_classifier_real= real_classifier_loss
        
            
            
            fake_img = G(z, fake_tag).to(device) 
            fake_score, _, _, _, _ = D(fake_img) #you need detach() here
        #input()
            #print(real_score)
            #print(real_score.shape) #torch.Size([64])
            #print(real_score.mean(0)) #tensor(-0.0618, device='cuda:0', grad_fn=<MeanBackward0>)
            #print(real_score.mean(0).shape) #torch.Size([])
            #input()
            fake_discrim_loss = fake_score.mean(0) #-log(1-P(S=fake|x_fake))
            #print(errD_fake) 
        #The classification loss for the synthesized image was omitted, since we believe the fake image would confuse the discriminator.
        #fake_classifier_loss= criterion(fake_predict, fake_tag)
             # Train with gradient penalty
            #print(real_img.data) #tensor
            #print(fake_img.data) #tensor
            #print(real_img.data.shape) #tensor #torch.Size([64, 3, 128, 128])
            #print(fake_img.data.shape) #tensor #torch.Size([64, 3, 128, 128])
            #input()
            gradient_penalty = calculate_gradient_penalty(real_img.data, fake_img.data, batch_size, D, device)
            #print(gradient_penalty)
            disc_cost= fake_discrim_loss- real_discrim_loss+ gradient_penalty
            classifier_loss= errD_classifier_real
            Wasserstein_D=  fake_discrim_loss- real_discrim_loss
            (disc_cost + classifier_loss).backward()  
            D_loss= disc_cost + classifier_loss
            classifier_log.append(classifier_loss.item())
            
        #D_optim.zero_grad() #location of zero_grad is strange
            D_optim.step()

        # Train generator
        for p in D.parameters():
            p.requires_grad= False
        #update G network
        G.zero_grad()
        z_g = torch.randn(batch_size, latent_dim).to(device)
        fake_tag_g = get_random_label(batch_size = batch_size, 
                                    hair_classes = hair_classes, hair_prior = None,
                                    eye_classes = eye_classes, eye_prior = None,
                                    face_classes= face_classes, face_prior= None,
                                    glass_classes= glass_classes, glass_prior=None).to(device)
                                    
        fake_hair_tags= fake_tag_g[: , 0:6]
        fake_eye_tags=fake_tag_g[:, 6:10]
        fake_face_tags= fake_tag_g[:, 10:13]
        fake_glass_tags= fake_tag_g[:, 13:15]
        #you forgot to do this, hope this will work!!!        
        z_g.requires_grad= True
        
        fake_img_g = G(z_g, fake_tag_g).to(device) #you don't have to generate two fake images
        
        
        fake_score, hair_predict, eye_predict, face_predict, glass_predict = D(fake_img_g)
        #print(fake_score)
        #print(fake_score.shape) #torch.Size([64])
        #print(fake_score.mean())
        #print(fake_score.mean().shape) #torch.Size([])
#tensor(-7.1275, device='cuda:0', grad_fn=<MeanBackward1>)

        #print(fake_score.mean().mean(0))
        #print(fake_score.mean().mean(0).shape)
        #input()

        discrim_loss_G = fake_score.mean() #discriminate whether the generated image is real or fake
        #-log(log(P(S=fake|X_fake)))
        #hair_classifier_loss_G = criterion(hair_predict, fake_hair_tags).mean()
        #eye_classifier_loss_G = criterion(eye_predict, fake_eye_tags).mean()
        #face_classifier_loss_G = criterion(face_predict, fake_face_tags).mean()
        #glass_classifier_loss_G = criterion(glass_predict, fake_glass_tags).mean()
        hair_classifier_loss_G = criterion(hair_predict, torch.max(fake_hair_tags, 1)[1]).mean()
        eye_classifier_loss_G = criterion(eye_predict, torch.max(fake_eye_tags, 1)[1]).mean()
        face_classifier_loss_G = criterion(face_predict, torch.max(fake_face_tags, 1)[1]).mean()
        glass_classifier_loss_G = criterion(glass_predict,torch.max(fake_glass_tags, 1)[1]).mean()
        classifier_loss_G = hair_classifier_loss_G +eye_classifier_loss_G + face_classifier_loss_G+ glass_classifier_loss_G
        #-log(P(C=c|X_fake))
        #here the loss function has been modified(both G and D), different from the original ACGAN paper
        gen_cost= -discrim_loss_G
        G_loss = classifier_loss_G + gen_cost
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
           # save_model(model = D, optimizer = D_optim, step = step_i, log = tuple(d_log), 
           #            file_path = os.path.join(checkpoint_dir, 'D_{}.ckpt'.format(step_i)))

            plot_loss(g_log = g_log, d_log = d_log, file_path = os.path.join(checkpoint_dir, 'loss.png'))
            plot_classifier_loss(log = classifier_log, file_path = os.path.join(checkpoint_dir, 'classifier loss.png'))

            generation_by_attributes(model = G, device = args.device,batch_size=batch_size ,step = step_i, latent_dim = latent_dim, 
                                     hair_classes = hair_classes, eye_classes = eye_classes, face_classes= face_classes, glass_classes=glass_classes,
                                     sample_dir = fixed_attribute_dir) # -->fix this later
    
if __name__ == '__main__':
    main()
            

        

    
    
    
