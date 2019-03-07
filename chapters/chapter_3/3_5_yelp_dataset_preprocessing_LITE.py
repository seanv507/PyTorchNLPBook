#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'chapters/chapter_3'))
	print(os.getcwd())
except:
	pass

#%%
import collections
import numpy as np
import pandas as pd
import re

from argparse import Namespace


#%%
args = Namespace(
    raw_train_dataset_csv="data/yelp/raw_train.csv",
    raw_test_dataset_csv="data/yelp/raw_test.csv",
    proportion_subset_of_train=0.1,
    train_proportion=0.7,
    val_proportion=0.15,
    test_proportion=0.15,
    output_munged_csv="data/yelp/reviews_with_splits_lite.csv",
    seed=1337
)


#%%
# Read raw data
train_reviews = pd.read_csv(args.raw_train_dataset_csv, header=None, names=['rating', 'review'])


#%%
# making the subset equal across the review classes
by_rating = collections.defaultdict(list)
for _, row in train_reviews.iterrows():
    by_rating[row.rating].append(row.to_dict())
    
review_subset = []

for _, item_list in sorted(by_rating.items()):

    n_total = len(item_list)
    n_subset = int(args.proportion_subset_of_train * n_total)
    review_subset.extend(item_list[:n_subset])

review_subset = pd.DataFrame(review_subset)


#%%
review_subset.head()


#%%
train_reviews.rating.value_counts()


#%%
review_subset.rating.value_counts()


#%%
# Unique classes
set(review_subset.rating)


#%%
# Splitting the subset by rating to create our new train, val, and test splits
by_rating = collections.defaultdict(list)
for _, row in review_subset.iterrows():
    by_rating[row.rating].append(row.to_dict())
    
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


#%%
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


#%%
# Write split data to file
final_reviews = pd.DataFrame(final_list)


#%%
final_reviews.split.value_counts()


#%%
# Preprocess the reviews
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"([.,!?])", r" \1 ", text)
    text = re.sub(r"[^a-zA-Z.,!?]+", r" ", text)
    return text
    
final_reviews.review = final_reviews.review.apply(preprocess_text)


#%%
final_reviews['rating'] = final_reviews.rating.apply({1: 'negative', 2: 'positive'}.get)


#%%
final_reviews.head()


#%%
final_reviews.to_csv(args.output_munged_csv, index=False)


