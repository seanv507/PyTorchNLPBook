#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 22:34:45 2019

@author: sviolante
"""
#%%
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

import spacy

#%%
# python -m spacy download en
nlp = spacy.load('en')

def describe(x):
        print("Type: {}".format(x.type()))
        print("Shape: {}".format(x.shape))
        print("Values: \n{}".format(x))


#%%
describe(torch.Tensor(2, 3))
describe(torch.rand(2,3))
describe(torch.randn(2,3))

x = torch.ones(2,3)
x = torch.fill_(x, 5)
#pytorch x_ means inplace operation


x = torch.Tensor([[1, 2, 3],
                  [4, 5, 6]])
# torch defaults to floats not doubles
x = torch.FloatTensor([[1, 2, 3],
                       [4, 5, 6]])

x = x.long()

# you can use add normal operators or torch.add() etc

x = torch.arange(6)

x = x.view(2,3) # reshape
torch.sum(x, dim=0)

torch.transpose(x, 0, 1)
describe(x[:1, :2])

describe(x[0, 1])

#complex indices

indices = torch.LongTensor([0, 2])
describe(torch.index_select(x, dim=1, index=indices))
# nb indices have to be longtensors
indices = torch.LongTensor([0, 0]) # so just repeats 0 row
describe(torch.index_select(x, dim=0, index=indices))

# concat
describe(torch.cat([x, x], dim=1))

describe(torch.stack([x, x]))


#matrix multiply
torch.mm(x, torch.transpose(x, 0, 1))


#requires_grad
x = torch.ones(2, 2, requires_grad=True)
describe(x)
print(x.grad is None)

y = (x + 2) * (x + 5) + 3
z = y.mean()
z.backward()
print(x.grad is None)

# handle cuda switch
print (torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


x = torch.rand(3, 3).to(device)
describe(x)

# for multi gpu environments
#  CUDA_VISIBLE_DEVICES= 0, 1, 2, 3 python main.py


# exercises
#%%
# create 2d tensor, then add dimension of size 1 inserted at dimension 0

x = torch.arange(6).view(2,3)[None, :, :] # was unsqueeze
describe(x)
# remove the extra dimension
x = x.squeeze(0)
describe(x)
#create random dim size(5,3) range [3,7)
x = torch.rand(5,3) * 4 + 3

#create random norm tensor

x = torch.randn(5,2)

#retrieve teh indexes of all the non zero elements in tensor
# torch.Tensor([1, 1, 1, 0,1])
x = torch.Tensor([1, 1, 1, 0,1])
ind = torch.nonzero(x )
describe(ind)

#create rand tensor opf size 3,1, then horizontally stack four copies together

z = torch.rand(3, 1)
# z1 = torch.hstack([z, z , z, z]) ?
z.expand(3, 4)

# return batch matrix-matrix product of 2 3d matrices
a = torch.rand(3, 4, 5)
b = torch.rand(3, 5, 4)

c = torch.einsum('bij,bjk->bik', a, b)
c = torch.bmm(a, b)

#%%
# chapter 2


text = "Mary, don't slap the green witch"
cln = [str(token) for token in nlp(text.lower())]
print(cln)

def n_grams(text, n):
    '''
    takes tokens or text, returns a list of n-grams
    '''
    return [text[i:i+n] for i in range(len(text)-n+1)]

#lemmatization
doc = nlp(u"he was running late")
for token in doc:
    print('{} --> {}'.format(token, token.lemma_))
print()

doc = nlp(u"Mary slapped the green witch.")
for token in doc:
    print('{} - {}'.format(token, token.pos_))

doc  = nlp(u"Mary slapped the green witch.")
for chunk in doc.noun_chunks:
    print ('{} - {}'.format(chunk, chunk.label_))

#%%
class Perceptron(nn.Module):
    """ A perceptron is one linear layer """
    def __init__(self, input_dim):
        """
        Args:
            input_dim (int): size of the input features
        """
        super(Perceptron, self).__init__()
        self.fc1 = nn.Linear(input_dim, 1)
       
    def forward(self, x_in):
        """The forward pass of the perceptron
        
        Args:
            x_in (torch.Tensor): an input data tensor 
                x_in.shape should be (batch, num_features)
        Returns:
            the resulting tensor. tensor.shape should be (batch,).
        """
        return torch.sigmoid(self.fc1(x_in)).squeeze()



#nb tanh sigmoid in torch not nn submodule
relu = nn.ReLU()
x = torch.range(-5., 5., 0.1)
y = relu(x)

plt.plot(x.numpy(), y.numpy())
plt.show()

#%%

mse_loss = nn.MSELoss()
outputs = torch.randn(3, 5, requires_grad=True)
targets = torch.randn(3, 5)
loss = mse_loss(outputs, targets)
print(loss)

#%%
# ce loss takes weighted sum as input 
ce_loss = nn.CrossEntropyLoss()
outputs = torch.randn(3, 5, requires_grad=True)
targets = torch.tensor([1, 0, 3], dtype=torch.int64)
loss = ce_loss(outputs, targets)
print(loss)
#%%
# but BCE doesn't !?
bce_loss = nn.BCELoss()
sigmoid = nn.Sigmoid()
probabilities = sigmoid(torch.randn(4, 1, requires_grad=True))
# int vs float targets?
targets = torch.tensor([1, 0, 1, 0],  dtype=torch.float32).view(4, 1)
loss = bce_loss(probabilities, targets)
print(probabilities)
print(loss)

#%%
input_dim = 2
lr = 0.001

perceptron = Perceptron(input_dim=input_dim)
bce_loss = nn.BCELoss()
optimizer = optim.Adam(params=perceptron.parameters(), lr=lr)

#%%
n_epochs = 5
n_batches = 
# each epoch is a complete pass over the training data
for epoch_i in range(n_epochs):
    # the inner loop is over the batches in the dataset
    for batch_i in range(n_batches):

        # Step 0: Get the data
        x_data, y_target = get_toy_data(batch_size)

        # Step 1: Clear the gradients 
        perceptron.zero_grad()

        # Step 2: Compute the forward pass of the model
        y_pred = perceptron(x_data, apply_sigmoid=True)

        # Step 3: Compute the loss value that we wish to optimize
        loss = bce_loss(y_pred, y_target)

        # Step 4: Propagate the loss signal backward
        loss.backward()

        # Step 5: Trigger the optimizer to perform one update
        optimizer.step()


#%%
# Split the subset by rating to create new train, val, and test splits
import collections
import pandas as pd

by_rating = collections.defaultdict(list)
for _, row in review_subset.iterrows():
    by_rating[row.rating].append(row.to_dict())

# Create split data
final_list = []
np.random.seed(args.seed)

for _, item_list in sorted(by_rating.items()):
    np.random.shuffle(item_list)
    
    n_total = len(item_list)
    n_train = int(args.train_proportion * n_total)
    n_val = int(args.val_proportion * n_total)
    n_test = int(args.test_proportion * n_total)
    
    # Give data point a split attribute
    for item in item_list[:n_train]:
        item['split'] = 'train'
    
    for item in item_list[n_train:n_train+n_val]:
        item['split'] = 'val'

    for item in item_list[n_train+n_val:n_train+n_val+n_test]:
        item['split'] = 'test'

    # Add to final list
    final_list.extend(item_list)

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"([.,!?])", r" \1 ", text)
    text = re.sub(r"[^a-zA-Z.,!?]+", r" ", text)
    return text


final_reviews = pd.DataFrame(final_list)


final_reviews.review = final_reviews.review.apply(preprocess_text)

#%%
import torch.nn as nn
import torch.nn.functional as F

class MultilayerPerceptron(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        Args:
            input_dim (int): the size of the input vectors
            hidden_dim (int): the output size of the first Linear layer
            output_dim (int): the output size of the second Linear layer
        """
        super(MultilayerPerceptron, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x_in, apply_softmax=False):
        """The forward pass of the MLP
        
        Args:
            x_in (torch.Tensor): an input data tensor 
                x_in.shape should be (batch, input_dim)
            apply_softmax (bool): a flag for the softmax activation
                should be false if used with the cross-entropy losses
        Returns:
            the resulting tensor. tensor.shape should be (batch, output_dim)
        """
        intermediate = F.relu(self.fc1(x_in))
        output = self.fc2(intermediate)
        
        if apply_softmax:
            output = F.softmax(output, dim=1)
        return output


#%%
batch_size = 2 # number of samples input at once
input_dim = 3
hidden_dim = 100
output_dim = 4

# Initialize model
mlp = MultilayerPerceptron(input_dim, hidden_dim, output_dim)
print(mlp)

#%%
def describe(x):
    print("Type: {}".format(x.type()))
    print("Shape/size: {}".format(x.shape))
    print("Values: \n{}".format(x))

x_input = torch.rand(batch_size, input_dim)
describe(x_input)

y_output = mlp(x_input, apply_softmax=False)
describe(y_output)

#%%
