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
from sklearn.cluster import KMeans
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def dataset_spliter(input_pickle, savingPath_train, savingPath_test):
    df = pd.read_pickle(input_pickle)
    df_sorted = df.sort_values(by='time_published')
    train_set, test_set = train_test_split(df_sorted, test_size=0.1, shuffle=False)
    train_set.to_pickle(savingPath_train)

    time_first = train_set.iloc[0]['time_published']

    test_set.to_pickle(savingPath_test)

    time_last = test_set.iloc[-1]['time_published']

    return time_first, time_last

def hashtag_threshold_filter(hashtags):
    threshold = 2000
    filtered = {k: set(v.keys()) for k, v in hashtags.items() if sum(v.values()) > threshold}
    loopingTimes = 0
    while len(filtered.keys()) < 100 or len(filtered.keys()) > 300:
        filtered_count = len(filtered.keys())
        if filtered_count < 100:
            threshold = int(threshold/2)
        else:
            threshold += 200
        loopingTimes += 1
        filtered = {k: set(v.keys()) for k, v in hashtags.items() if sum(v.values()) > threshold}

        if loopingTimes >= 200 or threshold == 0:
            raise ValueError (f"Warning: Threshold Adjust to {threshold}, still cant find hashtags in range, consider adjust lower/upper limit")

    print(f"{len(filtered.keys())} hashtags are selected")
    return filtered


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

def get_embedding(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128)
    outputs = model(**inputs)

    # Get the embeddings
    with torch.no_grad():
        outputs = model(**inputs)
    # Use the pooled output for a single embedding per input text
    return outputs.pooler_output

def clusterHashtags(hashtag_user_map_train, savingPath):
    tokenizer = AutoTokenizer.from_pretrained('Twitter/twhin-bert-base')
    model = AutoModel.from_pretrained('Twitter/twhin-bert-base')
    hashtag_vector = []
    for hashtag in tqdm(hashtag_user_map_train):
        hashtag_vector.append(get_embedding(hashtag, tokenizer, model))

    numpy_embeddings_list = [tensor.detach().cpu().numpy() for tensor in hashtag_vector]
    numpy_embeddings = np.stack(numpy_embeddings_list)

    np.save('/home/dayouwu2/event_detection_Wally/hashtag_evaluation/Hamas_Isreal_Testing/embeddings.npy', numpy_embeddings)

    loaded_embeddings = np.load('/home/dayouwu2/event_detection_Wally/hashtag_evaluation/Hamas_Isreal_Testing/embeddings.npy')
    reshaped_array = np.squeeze(loaded_embeddings, axis=1)
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

# def compute_similarity(hashtag1, hashtag2):
#     emb1 = get_embedding(hashtag1)
#     emb2 = get_embedding(hashtag2)

#     cosine_sim = cosine_similarity(emb1, emb2)
#     return cosine_sim

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

def cluster_time_map_to_train(cluster, time_first, time_last):
    time_gap = time_last - time_first + 1
    result = np.zeros(13)
    for timestamp in cluster:
        time_diff = timestamp - time_first
        timeId = time_diff * 13 / time_gap
        result[int(timeId)] += 1
    result = np.where(result > 0, 1, 0)
    return result

def logReg_dataset_prep(user_cluster_time_map_path, time_first, time_last, x_save, y_save):
    with open(user_cluster_time_map_path, 'r') as f:
        user_cluster_time_map = json.load(f)
    
    x_train = np.zeros((len(user_cluster_time_map) * 50, 12))
    y_train = np.zeros(len(user_cluster_time_map) * 50)
    index = -1
    for user in tqdm(user_cluster_time_map):
        index += 1
        for cluster_id in user_cluster_time_map[user]:
            cluster = user_cluster_time_map[user][cluster_id]
            array = cluster_time_map_to_train(cluster, time_first, time_last)
            row_id = int(cluster_id) + 50*index
            x_train[row_id] = array[:12]
            y_train[row_id] = np.max(array[12:])
    
    np.save(x_save, x_train)
    np.save(y_save, y_train)
    pass

def logReg_classifier(x_train, y_train, x_test, y_test):
    model = LogisticRegression()

    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"accuracy: {accuracy}")


if __name__ == "__main__":
    savingPath_train_x = "/home/dayouwu2/event_detection_Wally/hashtag_evaluation/Hamas_Isreal_Testing/hashtag_eval_train.pkl"
    savingPath_train_y = "/home/dayouwu2/event_detection_Wally/hashtag_evaluation/Hamas_Isreal_Testing/hashtag_eval_test.pkl"
    time_first, time_last = dataset_spliter("/home/dayouwu2/event_detection/event_detection_twitter/artifact_dir/temp_output/input_file.pkl", savingPath_train_x, savingPath_train_y)

    user_time_cluster_engagement_generator("/home/dayouwu2/event_detection/event_detection_twitter/artifact_dir/temp_output/input_file.pkl", "/home/dayouwu2/event_detection_Wally/hashtag_evaluation/hashtag_clusters.json", "/home/dayouwu2/event_detection_Wally/hashtag_evaluation/Hamas_Isreal_Testing/user_time_cluster_map.json")

    x_save = "/home/dayouwu2/event_detection_Wally/hashtag_evaluation/Hamas_Isreal_Testing/x_train.npy"
    y_save = "/home/dayouwu2/event_detection_Wally/hashtag_evaluation/Hamas_Isreal_Testing/y_train.npy"
    logReg_dataset_prep("/home/dayouwu2/event_detection_Wally/hashtag_evaluation/Hamas_Isreal_Testing/user_time_cluster_map.json", time_first, time_last, x_save, y_save)

    savingPath_test_x = "/home/dayouwu2/event_detection_Wally/hashtag_evaluation/french_election_testing/hashtag_eval_train.pkl"
    savingPath_test_y = "/home/dayouwu2/event_detection_Wally/hashtag_evaluation/french_election_testing/hashtag_eval_test.pkl"
    time_first_, time_last_ = dataset_spliter("/home/dayouwu2/event_detection_Wally/hashtag_evaluation/french_election_testing/input_file.pkl", savingPath_test_x, savingPath_test_y)
    
    hashtag_time_train, hashtag_user_map = hashtag_user_time_map_generator("/home/dayouwu2/event_detection_Wally/hashtag_evaluation/french_election_testing/input_file.pkl")

    filtered = hashtag_threshold_filter(hashtag_user_map)

    clusterHashtags(filtered, "/home/dayouwu2/event_detection_Wally/hashtag_evaluation/french_election_testing/hashtag_clusters.json")


    user_time_cluster_engagement_generator("/home/dayouwu2/event_detection_Wally/hashtag_evaluation/Hamas_Isreal_Testing/hashtag_eval_train.pkl", "/home/dayouwu2/event_detection_Wally/hashtag_evaluation/hashtag_clusters.json", "/home/dayouwu2/event_detection_Wally/hashtag_evaluation/Hamas_Isreal_Testing/user_time_cluster_map.json")
    
