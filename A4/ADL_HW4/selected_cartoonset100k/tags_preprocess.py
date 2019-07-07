import numpy as np
import torch
import pickle
f= open("cartoon_attr.txt", 'r')
print(f.readline())
print(f.readline())
image_to_tag={}
#results=[]
for line in f:
    #print(line)
    #print(line.split())
    line= line.split()
    #print(line[0])
    tags= list(map(int, line[1:]))
    #print(tags)
    #print(len(tags))
    #print(np.asarray(tags))
    image_to_tag[line[0]]= torch.from_numpy(np.asarray(tags))
    #split the line and then make a dictionary
    #results.append(line)
    #print(results)
    #image_to_tag[line[0]]: line[1:]
    #print(image_to_tag)
    #input()
#print(image_to_tag) 
#dump image_to_tag to pickle file
file_= open("image_to_tag.pickle", 'wb')
pickle.dump(image_to_tag, file_)
file_.close()
f.close()
