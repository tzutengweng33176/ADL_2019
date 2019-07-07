import torch
import torch.nn as nn

    
class Generator(nn.Module):
    """ ACGAN generator.
    
    ACGAN generator is simply a DCGAN generator that takes a noise vector and 
    class vector concatenated as input. All other details (activation functions,
    batch norm) follow the 2016 DCGAN paper.
    Attributes:
        latent_dim: the length of the noise vector
        class_dim: the length of the class vector (in one-hot form)
        gen: the main generator structure
    """


    def __init__(self, latent_dim, class_dim):
        """ Initializes Generator Class with latent_dim and class_dim."""
        super(Generator, self).__init__()
        
        self.latent_dim = latent_dim
        self.class_dim = class_dim
        #how to change output image size?????
        #https://github.com/pytorch/examples/issues/70  
        self.gen = nn.Sequential(
                    nn.ConvTranspose2d(in_channels = self.latent_dim + 
                    								 self.class_dim, 
                                       out_channels = 1024, 
                                       kernel_size = 4,
                                       stride = 1,
                                       bias = False),
                    nn.BatchNorm2d(1024),
                    nn.ReLU(inplace = True),
                    #state size: 1024*4*4
                    nn.ConvTranspose2d(in_channels = 1024,
                                       out_channels = 512,
                                       kernel_size = 4,
                                       stride = 2,
                                       padding = 1,
                                       bias = False),
                    nn.BatchNorm2d(512),
                    nn.ReLU(inplace = True),
                    #512*8*8
                    nn.ConvTranspose2d(in_channels = 512,
                                       out_channels = 256,
                                       kernel_size = 4,
                                       stride = 2,
                                       padding = 1,
                                       bias = False),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace = True),
                    #256*16*16
                    nn.ConvTranspose2d(in_channels = 256,
                                       out_channels = 128,
                                       kernel_size = 4,
                                       stride = 2,
                                       padding = 1,
                                       bias = False),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace = True),
                    #128*32*32
                    nn.ConvTranspose2d(in_channels = 128,
                                       out_channels = 64,
                                       kernel_size = 4,
                                       stride = 2,
                                       padding = 1,
                                       bias = False),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace = True),
                    #64*64*64
                    nn.ConvTranspose2d(in_channels = 64,
                                       out_channels = 3,
                                       kernel_size = 4,
                                       stride = 2,
                                       padding = 1),
                    nn.Tanh()
                    #3*128*128
                    )
        return
    
    def forward(self, _input, _class):
        """ Defines the forward pass of the Generator Class.
        Args:
            _input: the input noise vector.
            _class: the input class vector. The vector need not be one-hot 
                    since multilabel generation is supported.
        
        Returns:
            The generator output.
        """


        concat = torch.cat((_input, _class), dim = 1)  # Concatenate noise and class vector.
        #print(concat.shape) #torch.Size([128, 115])
        concat = concat.unsqueeze(2).unsqueeze(3)   # Reshape the latent vector into a feature map.
        #print(concat.shape) #torch.Size([128, 115, 1, 1])
        #input()
        return self.gen(concat)

class Discriminator(nn.Module):
    """ ACGAN discriminator.
    
    A modified version of the DCGAN discriminator. Aside from a discriminator
    output, DCGAN discriminator also classifies the class of the input image 
    using a fully-connected layer.

    Attributes:
    	num_classes: number of classes the discriminator needs to classify.
    	conv_layers: all convolutional layers before the last DCGAN layer. 
    				 This can be viewed as an feature extractor.
    	discriminator_layer: last layer of DCGAN. Outputs a single scalar.
    	bottleneck: Layer before classifier_layer.
    	classifier_layer: fully conneceted layer for multilabel classifiction.
			
    """
    def __init__(self,hair_classes ,eye_classes, face_classes, glass_classes):
        """ Initialize Discriminator Class with num_classes."""
        super(Discriminator, self).__init__()

        self.hair_classes = hair_classes
        self.eye_classes = eye_classes
        self.face_classes = face_classes
        self.glass_classes = glass_classes

        self.conv_layers = nn.Sequential(
                    nn.Conv2d(in_channels = 3, 
                             out_channels = 64, 
                             kernel_size = 4,
                             stride = 2,
                             padding = 1,
                             bias = False),
                    nn.LeakyReLU(0.2, inplace = True),
                    #conv1
                    nn.Conv2d(in_channels = 64, 
                             out_channels = 128, 
                             kernel_size = 4,
                             stride = 2,
                             padding = 1,
                             bias = False),
                    #nn.BatchNorm2d(128),
                    nn.LeakyReLU(0.2, inplace = True),
                    #conv2
                    nn.Conv2d(in_channels = 128, 
                             out_channels = 256, 
                             kernel_size = 4,
                             stride = 2,
                             padding = 1,
                             bias = False),
                    #nn.BatchNorm2d(256),
                    nn.LeakyReLU(0.2, inplace = True),
                    #conv3
                    nn.Conv2d(in_channels = 256, 
                             out_channels = 512, 
                             kernel_size = 4,
                             stride = 2,
                             padding = 1,
                             bias = False),
                    #nn.BatchNorm2d(512),
                    nn.LeakyReLU(0.2, inplace = True),
                    #conv4
                    nn.Conv2d(in_channels = 512, 
                             out_channels = 1024, 
                             kernel_size = 4,
                             stride = 2,
                             padding = 1,
                             bias = False),
                    #nn.BatchNorm2d(1024),
                    nn.LeakyReLU(0.2, inplace = True)
                    #conv5
                    )   
        self.discriminator_layer = nn.Sequential(
                    nn.Conv2d(in_channels = 1024, 
                             out_channels = 1, 
                             kernel_size = 4,
                             stride = 1) #in pytorch, it has bias= False
                    #conv6
#The output of D is no longer a probability, we do not apply sigmoid at the output of D.
#                    nn.Sigmoid()
                    ) 
        self.bottleneck = nn.Sequential(
                    nn.Conv2d(in_channels = 1024,  #6th conv
                             out_channels = 1024, 
                             kernel_size = 4,
                             stride = 1),
                    #nn.BatchNorm2d(1024),
                    nn.LeakyReLU(0.2)
                    )
        self.hair_classifier_layer = nn.Sequential(
                    nn.Linear(1024, 128),
                    #nn.BatchNorm1d(128),
                    nn.LeakyReLU(0.2),
                    nn.Linear(128, self.hair_classes),

                    nn.LogSoftmax(dim=1)
                    )
        self.eye_classifier_layer = nn.Sequential(
                    nn.Linear(1024, 128),
                    #nn.BatchNorm1d(128),
                    nn.LeakyReLU(0.2),
                    nn.Linear(128, self.eye_classes),

                    nn.LogSoftmax(dim=1)
                    )
        self.face_classifier_layer = nn.Sequential(
                    nn.Linear(1024, 128),
                    #nn.BatchNorm1d(128),
                    nn.LeakyReLU(0.2),
                    nn.Linear(128, self.face_classes),

                    nn.LogSoftmax(dim=1)
                    )
        self.glass_classifier_layer = nn.Sequential(
                    nn.Linear(1024, 128),
                    #nn.BatchNorm1d(128),
                    nn.LeakyReLU(0.2),
                    nn.Linear(128, self.glass_classes),

                    nn.LogSoftmax(dim=1)
                    )

        return
    
    def forward(self, _input):
        """ Defines a forward pass of a discriminator.
        Args:
            _input: A batch of image tensors. Shape: N * 3 * 128 *128
        
        Returns:
            discrim_output: Value between 0-1 indicating real or fake. Shape: N * 1
            aux_output: Class scores for each class. Shape: N * num_classes
        """
        #print(_input.shape) #torch.Size([128, 3, 128, 128])
        features = self.conv_layers(_input)  
        #print(features.shape) #torch.Size([128, 1024, 4, 4])
        discrim_output = self.discriminator_layer(features).view(-1) # Single-value scalar 
        #print(discrim_output)
        #print(discrim_output.shape) #torch.Size([batch_size])
        #input()
        #discriminate this image is real or fake in probability
        flatten = self.bottleneck(features).squeeze()
        #print(flatten.shape) #torch.Size([128, 512])
        hair_class= self.hair_classifier_layer(flatten)
        eye_class= self.eye_classifier_layer(flatten)
        face_class= self.face_classifier_layer(flatten)
        glass_class= self.glass_classifier_layer(flatten)
        #print(aux_output.shape) #torch.Size([128, 256, 5, 5]) --> should be (128, 15) OK
        #classify the image to see which class it belongs to
        #input()
        return discrim_output, hair_class, eye_class, face_class, glass_class

if __name__ == '__main__':
    latent_dim = 100
    class_dim = 15
    batch_size = 5
    z = torch.randn(batch_size, latent_dim)
    c = torch.randn(batch_size, class_dim)
    
    G = Generator(latent_dim, class_dim)
    D = Discriminator(6, 4, 3, 2)
    o = G(z, c)
    print(o.shape) #torch.Size([5, 3, 128, 128])
    x, y, z, v, w = D(o)
    print(x.shape, y.shape, z.shape, v.shape, w.shape) #torch.Size([5]) torch.Size([5, 22])
