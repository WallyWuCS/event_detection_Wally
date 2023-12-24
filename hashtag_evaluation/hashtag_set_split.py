# Citation:
# @article{zhang2022twhin,
#   title={TwHIN-BERT: A Socially-Enriched Pre-trained Language Model for Multilingual Tweet Representations},
#   author={Zhang, Xinyang and Malkov, Yury and Florez, Omar and Park, Serim and McWilliams, Brian and Han, Jiawei and El-Kishky, Ahmed},
#   journal={arXiv preprint arXiv:2209.07562},
#   year={2022}
# }

import json
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split

def dataset_spliter(input_pickle):
    df = pd.read_pickle(input_pickle)
    df_sorted = df.sort_values(by='time_published')
    train_set, test_set = train_test_split(df_sorted, test_size=0.1, shuffle=False)
    train_set.to_pickle("/home/dayouwu2/event_detection_Wally/hashtag_evaluation/Hamas_Isreal_Testing/hashtag_eval_train.pkl")

    time_first = train_set.iloc[0]['time_published']

    # validate_set, test_set = train_test_split(test_df, test_size=0.5, random_state=42)
    # validate_set.to_pickle("/home/dayouwu2/event_detection_Wally/hashtag_evaluation/Hamas_Isreal_Testing/hashtag_eval_validate.pkl")

    # time_second = validate_set.iloc[0]['time_published']

    test_set.to_pickle("/home/dayouwu2/event_detection_Wally/hashtag_evaluation/Hamas_Isreal_Testing/hashtag_eval_test.pkl")
    
    time_second = test_set.iloc[0]['time_published']
    time_last = test_set.iloc[-1]['time_published']

    return time_first, time_second, time_last


def hashtag_user_time_map_generator(input_pickle):
    df = pd.read_pickle(input_pickle)
    hashtag_time_map = defaultdict(list)
    hashtag_user_map = defaultdict(list)
    for index, row in tqdm(df.iterrows()):
        rawTweet = row['rawTweet']
        hashtags = set(x for x in rawTweet.split() if x.startswith('#'))
        user = row['name']
        time = row['time_published']
        for hashtag in hashtags:
            hashtag_time_map[hashtag].append(time)
            hashtag_user_map[hashtag].append(user)
    return hashtag_time_map, hashtag_user_map

def user_time_cluster_engagement_generator(input_pickle, hashtag_clusters_path, savingPath):
    with open(hashtag_clusters_path, 'r') as f:
        hashtag_clusters = json.load(f)
    element_to_cluster = {element: cluster_id for cluster_id, elements in hashtag_clusters.items() for element in elements}

    df = pd.read_pickle(input_pickle)
    
    user_cluster_time_map = defaultdict(lambda: defaultdict(list))
    for index, row in tqdm(df.iterrows()):
        rawTweet = row['rawTweet']
        hashtags = set(x for x in rawTweet.split() if x.startswith('#'))
        user = row['name']
        time = row['time_published']
        for hashtag in hashtags:
            cluster_id = element_to_cluster.get(hashtag, None)
            if cluster_id:
                user_cluster_time_map[user][int(cluster_id)].append(time)
    
    with open(savingPath, 'w') as f:
        json.dump(user_cluster_time_map, f)
    
    pass

def get_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128)
    outputs = model(**inputs)

    # Get the embeddings
    with torch.no_grad():
        outputs = model(**inputs)
    # Use the pooled output for a single embedding per input text
    return outputs.pooler_output

def compute_similarity(hashtag1, hashtag2):
    emb1 = get_embedding(hashtag1)
    emb2 = get_embedding(hashtag2)

    cosine_sim = cosine_similarity(emb1, emb2)
    return cosine_sim

def clusterHashtags(hashtag_user_map_train, savingPath):
    hashtag_vector = []
    for hashtag in tqdm(hashtag_user_map_train):
        hashtag_vector.append(get_embedding(hashtag))

    numpy_embeddings_list = [tensor.detach().cpu().numpy() for tensor in hashtag_vector]
    numpy_embeddings = np.stack(numpy_embeddings_list)

    np.save('/home/dayouwu2/event_detection_Wally/hashtag_evaluation/Hamas_Isreal_Testing/embeddings.npy', numpy_embeddings)

    # tensor_data = torch.stack(normalized_tensors).view(len(normalized_tensors), -1).numpy()
    loaded_embeddings = np.load('/home/dayouwu2/event_detection_Wally/hashtag_evaluation/Hamas_Isreal_Testing/embeddings.npy')
    reshaped_array = np.squeeze(loaded_embeddings, axis=1)
    # similarity_matrix = cosine_similarity(reshaped_array)
    n_clusters = 50
    kmeans = KMeans(n_clusters = n_clusters, init='k-means++', random_state=42)
    kmeans.fit(reshaped_array)

    clusters = kmeans.labels_
    cluster_to_hashtags = defaultdict(list)
    for hashtag, cluster_label in zip(list(hashtag_user_map_train.keys()), clusters):
        cluster_to_hashtags[int(cluster_label)].append(hashtag)
    
    with open(savingPath, 'w') as f:
        json.dump(cluster_to_hashtags, f)
    pass

def cluster_time_map_to_train(cluster, time_first, time_last):
    time_gap = time_last - time_first + 1
    result = np.zeros(9)
    for timestamp in cluster:
        time_diff = timestamp - time_first
        timeId = time_diff * 9 / time_gap
        result[int(timeId)] += 1
    return result

def logReg_dataset_prep(user_cluster_time_map_path, time_first, time_last):
    with open(user_cluster_time_map_path, 'r') as f:
        user_cluster_time_map = json.load(f)
    
    x_train = np.zeros((len(user_cluster_time_map) * 50, 8))
    y_train = np.zeros(len(user_cluster_time_map) * 50)
    index = -1
    for user in tqdm(user_cluster_time_map):
        index += 1
        for cluster_id in user_cluster_time_map[user]:
            cluster = user_cluster_time_map[user][cluster_id]
            array = cluster_time_map_to_train(cluster, time_first, time_last)
            row_id = int(cluster_id) + 50*index
            x_train[row_id] = array[:8]
            if array[8] > 0:
                y_train[row_id] = 1
    
    np.save("x_train.npy", x_train)
    np.save("y_train.npy", y_train)
    pass




if __name__ == "__main__":
    time_first, time_second, time_last = dataset_spliter("/home/dayouwu2/event_detection/event_detection_twitter/artifact_dir/temp_output/input_file.pkl")

    tokenizer = AutoTokenizer.from_pretrained('Twitter/twhin-bert-base')
    model = AutoModel.from_pretrained('Twitter/twhin-bert-base')
    
    # hashtag_time_train, hashtag_user_map = hashtag_user_time_map_generator("/home/dayouwu2/event_detection/event_detection_twitter/artifact_dir/temp_output/input_file.pkl")
    # filtered_hashtag_user= {key: value for key, value in hashtag_user_map.items() if len(value) > 2000}
    # clusterHashtags(filtered_hashtag_user, "/home/dayouwu2/event_detection_Wally/hashtag_evaluation/hashtag_clusters.json")

    user_time_cluster_engagement_generator("/home/dayouwu2/event_detection_Wally/hashtag_evaluation/Hamas_Isreal_Testing/hashtag_eval_train.pkl", "/home/dayouwu2/event_detection_Wally/hashtag_evaluation/hashtag_clusters.json", "/home/dayouwu2/event_detection_Wally/hashtag_evaluation/Hamas_Isreal_Testing/user_time_cluster_map.json")
    
