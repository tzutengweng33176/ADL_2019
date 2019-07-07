#How to train my model?
'''
python3 train_WGANGP_split.py

latent_dim = 100
Iteration= 50000
batch_size= 64
Learning rate= 0.0001
Lambda= 10 for gradient penalty
Optimizer: Adam, beta= (0.5, 0.999)

Discriminator was trained 5 times more than the generator.
I have replaced real labels(all 1's) to smooth labels(0.7~1.0) for training.


'''

#How to plot the figures in my report?

'''
I use save_image function in torchvision.utils.
See train_WGANGP_split.py for details.
'''

