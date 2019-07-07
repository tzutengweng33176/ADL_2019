import pickle

with open('image_to_tag.pickle', 'rb') as file:
    dict_1=pickle.load(file)

print(dict_1) #test OK
