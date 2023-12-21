import json
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import train_test_split

def dataset_spliter(input_pickle):
    df = pd.read_pickle(input_pickle)
    df_sorted = df.sort_values(by='time_published')
    train_set, test_set = train_test_split(df_sorted, test_size=0.1, random_state=42)
    train_set.to_pickle("event_detection_Wally/hashtag_evaluation/Hamas_Isreal_Testing/hashtag_eval_train.pkl")
    test_set.to_pickle("event_detection_Wally/hashtag_evaluation/Hamas_Isreal_Testing/hashtag_eval_test.pkl")

def hashtag_user_time_map_generator(input_pickle):
    df = pd.read_pickle(input_pickle)
    hashtag_time_map = defaultdict(list)
    hashtag_user_map = defaultdict(list)
    for index, row in df.iterrows():
        rawTweet = row['rawTweet']
        hashtags = set(x for x in rawTweet.split() if x.startswith('#'))
        user = row['name']
        time = row['time_published']
        for hashtag in hashtags:
            hashtag_time_map[hashtag].append(time)
            hashtag_user_map[hashtag].append(user)

if __name__ == "__main__":
    dataset_spliter("/home/dayouwu2/event_detection/event_detection_twitter/artifact_dir/temp_output/input_file.pkl")