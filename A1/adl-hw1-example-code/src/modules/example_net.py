import torch


class ExampleNet(torch.nn.Module):
    """

    Args:

    """

    def __init__(self, dim_embeddings,
                 similarity='inner_product'):
        super(ExampleNet, self).__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(dim_embeddings, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256)
        )

    def forward(self, context, context_lens, options, option_lens):
        #print("context before mlp:", context.shape) # torch.Size([10, 76, 200]); evaluation 300d:  torch.Size([10, 30, 300])
        #print('self.mlp(context): ', self.mlp(context).shape) #torch.Size([10, 29, 256])
        #print('self.mlp(context).max(1): ', self.mlp(context).max(1).shape)
        context = self.mlp(context).max(1)[0] #max(1):  max value of each row
        #input()
        #print("context after mlp: ", context.shape) #torch.Size([10, 256]);  evalutaion 300d: torch.Size([10, 256])
        logits = []
        #print("options before mlp: ", options.shape) #torch.Size([10, 5, 50, 200]); evaluation 300d: torch.Size([10, 5, 50, 300])
        for i, option in enumerate(options.transpose(1, 0)):
            #print("option before mlp: ", option.shape) #torch.Size([10, 44, 200]); evaluation 300d: torch.Size([10, 50, 300])
            option = self.mlp(option).max(1)[0] #https://pytorch.org/docs/stable/torch.html?highlight=max#torch.max
            #print("option after mlp: ", option.shape) #torch.Size([10, 256]); evaluation 300d: torch.Size([10, 256])
            #input()
            logit = ((context - option) ** 2).sum(-1) #sum(-1): axis may be negative, in which case it counts from the last to the first axis.
            #print("(context-option): ", (context-option))
            #print("(context-option)**2: ", (context - option) ** 2)
            #print("logit: ", logit)
            #print("logit.shape: ", logit.shape)  # torch.Size([10])
            logits.append(logit)
            #print("i= ", i) #i=0~4; evaluation i=0~4
        #print("logits before torch.stack: ",logits) # len(logits)=5
        logits = torch.stack(logits, 1) #Concatenates sequence of tensors along a new dimension. All tensors need to be of the same size.
        #print("logits after torch.stack: ", logits.shape) #torch.Size([10, 5])
        #input()
        return logits
