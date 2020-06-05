import os
import json
import pandas as pd
import pickle
import numpy as np


DOR_DIR = '/data'
TPS_DIR = 'data/music'
UNI_DIR = 'data/music/dmfcur'
TP_file = os.path.join(DOR_DIR, 'Music.json')
print("start load music!")


f = open(TP_file)
users_id=[]
items_id=[]
ratings=[]
reviews=[]
np.random.seed(2020)

for line in f:
    js=json.loads(line)
    if str(js['reviewerID'])=='unknown':
        print ("unknown")
        continue
    if str(js['asin'])=='unknown':
        print ("unknown2") 
        continue
    reviews.append(js['reviewText'])
    users_id.append(str(js['reviewerID'])+',')
    items_id.append(str(js['asin'])+',')
    ratings.append(str(js['overall']))
    data=pd.DataFrame({'user_id':pd.Series(users_id),
                   'item_id':pd.Series(items_id),
                   'ratings':pd.Series(ratings),
                   'reviews':pd.Series(reviews)})[['user_id','item_id','ratings','reviews']]

def get_count(tp, id):
    playcount_groupbyid = tp[[id, 'ratings']].groupby(id, as_index=False)
    count = playcount_groupbyid.size()
    return count
usercount, itemcount = get_count(data, 'user_id'), get_count(data, 'item_id')

unique_uid = usercount.index
unique_sid = itemcount.index

item2id = dict((sid, i) for (i, sid) in enumerate(unique_sid))
user2id = dict((uid, i) for (i, uid) in enumerate(unique_uid))

def numerize(tp):
    uid = list(map(lambda x: user2id[x], tp['user_id']))
    sid = list(map(lambda x: item2id[x], tp['item_id']))
    tp['user_id'] = uid
    tp['item_id'] = sid
    return tp

data=numerize(data)
tp_rating=data[['user_id','item_id','ratings']]


user_nums = usercount.shape[0]
item_nums = itemcount.shape[0]

def creatMtrix( ):
    train_matrix = np.zeros([user_nums, item_nums], dtype=np.float32)
    for index, row in tp_rating.iterrows():
        userid = row[0]
        itemid = row[1]
        ratings = float(row[2])
        train_matrix[userid][itemid] = ratings
    return np.array(train_matrix)
rating_matrix = creatMtrix()


n_ratings = tp_rating.shape[0]
print("稀疏度： ", n_ratings/(user_nums*item_nums))
print("用户平均评论数： ", n_ratings/(user_nums))

test = np.random.choice(n_ratings, size=int(0.10 * n_ratings), replace=False)
test_idx = np.zeros(n_ratings, dtype=bool)
test_idx[test] = True

tp_valid = tp_rating[test_idx]
tp_train= tp_rating[~test_idx]
data2=data[test_idx]
data=data[~test_idx]


tp_train.to_csv(os.path.join(TPS_DIR, 'train.csv'), index=False,header=None)
tp_valid.to_csv(os.path.join(TPS_DIR, 'valid.csv'), index=False,header=None)


print("csv load done!")

user_reviews={}
item_reviews={}
user_rid={}
item_rid={}

for i in data.values:
    user_id = i[0]
    item_id = i[1]
    review_text = i[3]
    if user_id in user_reviews:
        user_reviews[user_id].append(review_text)
        user_rid[user_id].append(item_id)
    else:
        user_reviews[user_id] = [review_text]
        user_rid[user_id] = [item_id]
    if item_id in item_reviews:
        item_reviews[item_id].append(review_text)
        item_rid[item_id].append(user_id)
    else:
        item_reviews[item_id] = [review_text]
        item_rid[item_id] = [user_id]

for i in data2.values:
    if i[0] in user_reviews:
        l=1
    else:
        user_rid[i[0]]=[0]
        user_reviews[i[0]]=['0']
    if i[1] in user_reviews:
        l=1
    else:
        item_reviews[i[1]] = ['0']
        item_rid[i[1]]=[0]

pickle.dump(user_reviews, open(os.path.join(TPS_DIR, 'user_review'), 'wb'))
pickle.dump(item_reviews, open(os.path.join(TPS_DIR, 'item_review'), 'wb'))


pickle.dump(user_rid, open(os.path.join(TPS_DIR, 'user_rid'), 'wb'))
pickle.dump(item_rid, open(os.path.join(TPS_DIR, 'item_rid'), 'wb'))

#************************************************
pickle.dump(rating_matrix, open(os.path.join(UNI_DIR, 'rating_matrix'), 'wb'))
#************************************************

print("total user num:", user_nums, " ", "total item num:", item_nums)