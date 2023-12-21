import ast
import difflib
import json
import os
import random
import shutil
import subprocess
import time
from bisect import bisect_left
from collections import Counter, defaultdict
from datetime import datetime
from itertools import islice

import apollo2_api_client
import Levenshtein
import numpy as np
import openai
import pandas as pd
from tqdm import tqdm
from unidecode import unidecode

temp_output_dir = "temp_out"
def updateSavingFolder(path):
    global temp_output_dir
    temp_output_dir = path

##Function used to remove temp files
def delete_folder_and_contents(folder_path):
    # Check if the folder exists
    if os.path.exists(folder_path):
        # Remove the folder and all its contents
        shutil.rmtree(folder_path)
        print("Folder deleted")
    else:
        print("Folder doesnt exist, check path")

##Create a temp directory and save the initial processed file as input.json
def preprocess(input_filepath):
    def preprocess_input():
        df = pd.read_pickle(input_filepath)
        df= df.drop_duplicates(subset=['name', 'rawTweet'])
        df['rawTweet'] = df['rawTweet'].str.replace(r'https://\S+', '', regex=True)
        df['rawTweet_id'] = df.index.astype(str) + ' ' + df['rawTweet']

        selected_columns = df[['rawTweet_id', 'time_published']]
        selected_columns = selected_columns.rename(columns={'rawTweet_id': 'content', 'time_published': 'timestamp'})
        list_of_dicts = selected_columns.to_dict(orient='records')

        # Now JSON_File contains the desired format
        save_path = f'{temp_output_dir}/input.json'
        with open(save_path, 'w') as file:
            json.dump(list_of_dicts, file)

    preprocess_input()

##Run Jinyang's event detection package to detect the events, will require to install the package first, 
#  cr: https://github.com/jinyangustc/event_detection
def event_detection():
    def run_command(command):
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            print(f'Command failed with error: {stderr.decode()}')
        else:
            print(stdout.decode())

    venv_dir = 'venv'
    run_command(f'virtualenv {venv_dir}')

    activate_script_path = os.path.join(venv_dir, 'bin', 'activate')
    command_prefix = f'source {activate_script_path}; '

    run_command(f'{command_prefix}')
    run_command('pip install -e .')

    run_command(f'apollo stopwords -v -t 0.10 -i {temp_output_dir}/input.json -o {temp_output_dir}/stopwords.txt')
    run_command(f'apollo detect -c data/config.toml -i {temp_output_dir}/input.json -s {temp_output_dir}/stopwords.txt > {temp_output_dir}/timeline.txt')

##Process event detection results and save them to temp directory (raw_content.json and processed_events.json)
def postprocess_events(input_filepath):
    window_times = []
    events_pair = []
    contents = []
    ids_window = []
    contents_window = []
    capture = False
    oneContent = []

    def getUserId(filepath, tweet_id):
        data = pd.read_pickle(filepath)
        userIds = []
        for ids in tqdm(tweet_id):
            usersEachTweet = []
            for id in ids:
                # print(id)
                usersEachTweet.append(data.loc[int(id), 'name'])
            userIds.append(usersEachTweet)
        return userIds
        
    def extractInfo(contents):
        for content in contents:
            gettingID = False
            gettingEvents = True
            gettingContents = False
            iDs = []
            content_in_a_window = []
            contentText = ""
            events = ""
            # events_pair.append(content[0])
            for index, line in enumerate(content):
                if len(line) == 0:
                    gettingID = True
                    gettingEvents = False
                    gettingContents = False
                    content_in_a_window.append(contentText)
                    contentText = ""
                    continue
                if gettingEvents:
                    events += line
                if gettingContents:
                    contentText += line
                if gettingID:
                    iDs.append(line.split(' ', 1)[0])
                    contentText += line.split(' ', 1)[1]
                    gettingContents = True
                    gettingID = False
                    
            ids_window.append(iDs)
            contents_window.append(content_in_a_window[1:])
            events_pair.append(events)

    def saveResult(window_times, events_pair, ids_window, user_id, savePath):
        if len(window_times) != len(events_pair):
            print("event length not same, correct this first\n")
            return False
        
        output_toSave = {}
        for index in range(len(window_times)):
            output_toSave[f'event{index}'] = {
                'window_times': window_times[index],
                'word_pairs': events_pair[index],
                'tweet_id_involved': ids_window[index],
                'userName_involved': user_id[index]
            }
        with open(savePath, 'w') as f:
            json.dump(output_toSave, f)
        
        return True

    #Handle original file into contents
    with open(f'{temp_output_dir}/timeline.txt', 'r') as file:
        lines = file.readlines()
        i = 0
        while i < len(lines):
            line = lines[i].strip()         

            if capture:
                if line.startswith("###"):
                    i += 1
                    continue
                if line.startswith("---"):
                    capture = False
                    contents.append(oneContent)
                    oneContent = []
                else:
                    oneContent.append(line)

            # Extract window time
            if line.startswith("window from") and lines[i + 2].startswith('['):
                window_times.append(line[12:])
                # Extract word pairs
                capture = True

            i += 1

    extractInfo(contents=contents)
    userId = getUserId(input_filepath, ids_window)
    userLen = []
    for users in userId:
        userLen.append(len(set(users)))

    # Print the extracted data
    sortedData = sorted(zip(userLen, window_times, events_pair, ids_window, userId, contents_window), reverse=True)
    userLen, window_times, events_pair, ids_window, userId, contents_window = map(list, zip(*sortedData))

    file_path_contents = f'{temp_output_dir}/raw_content.json'
    with open(file_path_contents, 'w') as f:
        json.dump(contents_window, f)

    saveResult(window_times=window_times, events_pair=events_pair, ids_window=ids_window, user_id=userId, savePath=f'{temp_output_dir}/processed_events.json')

##Multi-used function specially used for finding word pairs
def find_word_pair(word_pairs):
# Iterate over the word pairs
    for pair in word_pairs:
        # Check if both words in the pair do not start with '//t'
        if not pair[0].startswith('//t') and not pair[1].startswith('//t'):
            return ' '.join(pair)
    
    # If no pair found, check for pairs with one word not starting with '//t'
    for pair in word_pairs:
        if not pair[0].startswith('//t'):
            return ' |'.join(pair)
        else:
            if not pair[1].startswith('//t'):
                return pair[1] + ' |' + pair[0]    
        # If no such pair found, skip current function
    return None

##Merge events based on user amount
def merge_events(user_amount_threshold = 100):
    file_path_read = f'{temp_output_dir}/processed_events.json'
    with open(file_path_read, 'r') as f:
        events_data = json.load(f)

    currentUserLen = 0
    currentUsers = []
    window_time_list = []
    final_result = {}
    CurrentWordPair = ""

    for key in events_data:
        word_pairs = events_data[key]['word_pairs']
        word_pairs = ast.literal_eval(word_pairs)
        if find_word_pair(word_pairs):
            wordPair = find_word_pair(word_pairs)
        else:
            continue

        window_time = events_data[key]['window_times']
        date_str1, date_str2 = window_time.split(' to ')
        date1 = datetime.strptime(date_str1, '%Y-%m-%d %H:%M:%S')
        timestamp1 = int(time.mktime(date1.timetuple()))
        window_time_list.append(timestamp1)

        user_involved = events_data[key]['userName_involved']
        if currentUserLen == 0:
            currentUserLen = len(user_involved)
            currentUsers = user_involved
            CurrentWordPair = wordPair
        else:
            if currentUserLen != len(user_involved):
                if wordPair == CurrentWordPair:
                    continue
                final_result[CurrentWordPair] = {
                    'window_timestamp': window_time_list,
                    'usersInvolved': currentUsers
                }
                window_time_list = [timestamp1]
                currentUserLen = len(user_involved)
                currentUsers = user_involved
                CurrentWordPair = wordPair

    update_result = {}
    for key in final_result:
        userLen = len(final_result[key]['usersInvolved'])
        if userLen >= user_amount_threshold:
            update_result[key] = final_result[key]

    file_path_write = f'{temp_output_dir}/merged_events_caped.json'
    with open(file_path_write, 'w') as f:
        json.dump(update_result, f)

##Get user delta value and reorder the events
def getUserDelta(input_filepath):
    def median(nums):
        sorted_nums = sorted(nums)
        n = len(sorted_nums)
        
        # If even, return the average of the two middle numbers
        if n % 2 == 0:
            return (sorted_nums[n // 2 - 1] + sorted_nums[n // 2]) / 2
        # If odd, return the middle number
        else:
            return sorted_nums[n // 2]
    
    def makeUserMessageMap(filepath):
        data = pd.read_pickle(filepath)
        user_message_map = {}
        userName_list = data['name'].to_list()
        for userName in userName_list:
            if userName not in user_message_map:
                user_message_map[userName] = 1
            else:
                user_message_map[userName] += 1
        return user_message_map
    
    user_message_map = makeUserMessageMap(input_filepath)
    with open(f'{temp_output_dir}/user_message_map.json', 'w') as f:
        json.dump(user_message_map, f)

    with open(f'{temp_output_dir}/merged_events_caped.json', 'r') as f:
        merged_events = json.load(f)

    event_userMessage_map = {}
    user_involved_total = []

    for event in merged_events:
        users_involved = merged_events[event]['usersInvolved']
        user_involved_total += users_involved
        event_userMessage_map[event] = {}
        event_userMessage_map[event]['users'] = users_involved
        event_userMessage_map[event]['user_messages_count'] = []
        for user in users_involved:
            user_messages = user_message_map[user]
            event_userMessage_map[event]['user_messages_count'].append(user_messages)

    user_involved_total = list(set(user_involved_total))

    for event in event_userMessage_map:
        user_messages = event_userMessage_map[event]['user_messages_count']
        user_involved = event_userMessage_map[event]['users']
        user_not_involved = [x for x in user_involved_total if x not in user_involved]

        user_not_messages = []
        for user in user_not_involved:
            message_count = user_message_map[user]
            user_not_messages.append(message_count)
        
        in_median = median(user_messages)
        no_median = median(user_not_messages)
        delta = in_median - no_median
        event_userMessage_map[event]['delta'] = delta
        merged_events[event]['delta'] = delta
    
    sorted_events = dict(sorted(merged_events.items(), key=lambda item:item[1]['delta'], reverse=True))
    first_50_events = dict(islice(sorted_events.items(), 50))
    with open(f'{temp_output_dir}/output1_delta.json', 'w') as f:
        json.dump(first_50_events, f)

##Add summarization to the selected events and save them all to output1_events.json
def addEventSummarization(input_filepath, openAI_key, folderName = f'{temp_output_dir}'):
    with open(f'{temp_output_dir}/processed_events.json', 'r') as f:
        processed_events = json.load(f)
    with open(f'{temp_output_dir}/output1_delta.json', 'r') as f:
        merged_events = json.load(f)
    
    event_tweets_map = {}
    for key in processed_events:
        word_pairs = processed_events[key]['word_pairs']
        word_pairs = ast.literal_eval(word_pairs)
        if find_word_pair(word_pairs):
            wordPair = find_word_pair(word_pairs)
            processed_events[key]['word_pairs'] = wordPair
            if wordPair in event_tweets_map.keys():
                event_tweets_map[wordPair] += processed_events[key]['tweet_id_involved']
            else:
                event_tweets_map[wordPair] = processed_events[key]['tweet_id_involved']
            event_tweets_map[wordPair] = list(set(event_tweets_map[wordPair]))
        else:
            continue

    openai_client = openai.OpenAI(api_key=openAI_key) if openAI_key else None

    def merge_similar_strings(strings, cutoff=0.8):
        """Merge similar strings in the list based on a similarity cutoff."""
        merged = []
        while strings:
            current_str = strings[0]
            # Find close matches to the current string in the list
            similar = difflib.get_close_matches(current_str, strings, n=len(strings), cutoff=cutoff)
            # Merge the similar strings into one, for this example, we just choose the first one
            merged.append(current_str)
            # Remove the similar strings from the original list
            for s in similar:
                strings.remove(s)
        return merged

    df = pd.read_pickle(input_filepath)
    raw_input_list = df.to_dict('records')
    texts_to_summarize = []
    summarize_result = []
    for key in merged_events:
        tweet_Ids = event_tweets_map[key]
        raw_contents = []
        for tweetId in tweet_Ids:
            tweetId = int(tweetId)
            if raw_input_list[tweetId]['rawTweet']:
                raw_contents.append(raw_input_list[tweetId]['rawTweet'])
                if len(raw_contents) > 60:
                    break
        merged_strings = merge_similar_strings(raw_contents)
        text_to_summarize = "\n".join(merged_strings)
        texts_to_summarize.append(text_to_summarize)
        print(len(text_to_summarize))

        if openai_client is not None:
            prompt_text = "You are trying to help user summarize several twitter message into a very short sentence (less than 20 words)"
            message_text = f"Provide a very very brief and concise summary of the following text in one sentence(English), must somehow related to the word pair {key}:\n{text_to_summarize}\n"
            response = openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": prompt_text},
                    {"role": "user", "content": message_text}
                ]
            )
            summary = response.choices[0].message.content
        else:
            summary = "openai key not provided"
        summary = summary.replace("\r\n", " ").replace("\r", " ").replace("\n", " ")
        summarize_result.append(summary)
    
    for index, key in enumerate(list(merged_events.keys())):
        merged_events[key]['summarization'] = summarize_result[index]

    output1_events = {}
    for key in merged_events:
        unicode_key = unidecode(key)
        parts = unicode_key.split(maxsplit=1)
        unicode_key = "___".join(parts)
        output1_events[unicode_key] = merged_events[key]
        print(unicode_key)

    wordpair_summary_map = []
    for index, key in enumerate(list(output1_events.keys())):
        oneSummary = {}
        oneSummary['word_pair'] = key
        oneSummary['summarization'] = summarize_result[index]
        oneSummary['content'] = texts_to_summarize[index]
        wordpair_summary_map.append(oneSummary)

    df = pd.DataFrame(wordpair_summary_map)

    df.to_csv(f'{folderName}/wordpair_summary_map.csv', index=False, sep=',')
    df.to_excel(f'{folderName}/wordpair_summary_map.xlsx', index=False)

    with open(f'{folderName}/event_summary_map.txt', 'w') as f:
        for row in wordpair_summary_map:
            key = row['word_pair']
            item = row['summarization']
            f.write(f"{key}: {item}\n")

    with open(f'{folderName}/wordpair_summary_map.json', 'w') as f:
        json.dump(wordpair_summary_map, f)

    with open(f'{temp_output_dir}/output1_events.json', 'w') as f:
        json.dump(output1_events, f)

##Add summarization to the selected hashtags and save them all to output1_hashtags.json, need input your api_key and change other parameters
def addHashtagSummarization(input_filepath, openAI_key, hashtag_lower_limit=50, hashtag_upper_limit=200, hashtag_count_threshold = 200, instruction = "conflict", folderName = f'{temp_output_dir}'):
    df = pd.read_pickle(input_filepath)
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
    hashtag_user_map_dict = {k: dict(Counter(v).most_common()) for k, v in hashtag_user_map.items()}
    ##This threshold is the least amount of users involved in hashtags
    filtered = {k: set(v.keys()) for k, v in hashtag_user_map_dict.items() if sum(v.values()) > hashtag_count_threshold}

    loopingTimes = 0
    while len(filtered.keys()) < hashtag_lower_limit or len(filtered.keys()) > hashtag_upper_limit:
        filtered_count = len(filtered.keys())
        if filtered_count < hashtag_lower_limit:
            hashtag_count_threshold = int(hashtag_count_threshold/2)
        else:
            hashtag_count_threshold += 200
        loopingTimes += 1
        filtered = {k: set(v.keys()) for k, v in hashtag_user_map_dict.items() if sum(v.values()) > hashtag_count_threshold}

        if loopingTimes >= 200 or hashtag_count_threshold == 0:
            raise ValueError (f"Warning: Threshold Adjust to {hashtag_count_threshold}, still cant find hashtags in range, consider adjust lower/upper limit")

    print(f"Threshold has been adjust to: {len(filtered.keys())} for hashtag selection")

    with open(f'{temp_output_dir}/user_message_map.json', 'r') as f:
        user_message_map = json.load(f)

    def median(nums):
        sorted_nums = sorted(nums)
        n = len(sorted_nums)
        
        # If even, return the average of the two middle numbers
        if n % 2 == 0:
            return (sorted_nums[n // 2 - 1] + sorted_nums[n // 2]) / 2
        # If odd, return the middle number
        else:
            return sorted_nums[n // 2]
        
    hashtag_userMessage_map = {}
    user_involved_total = []

    for hashtag in filtered:
        users_involved = hashtag_user_map[hashtag]
        user_involved_total += users_involved
        hashtag_userMessage_map[hashtag] = {}
        hashtag_userMessage_map[hashtag]['users'] = users_involved
        hashtag_userMessage_map[hashtag]['user_messages_count'] = []
        for user in users_involved:
            user_messages = user_message_map[user]
            hashtag_userMessage_map[hashtag]['user_messages_count'].append(user_messages)

    user_total_set = set(user_involved_total)

    # Calculate hashtag delta value
    for hashtag in tqdm(hashtag_userMessage_map):
        user_messages = hashtag_userMessage_map[hashtag]['user_messages_count']
        user_involved = hashtag_userMessage_map[hashtag]['users']
        user_involved_set = set(user_involved)
        user_not_involved = list(user_total_set - user_involved_set)
        # user_not_involved = [x for x in user_involved_total if x not in user_involved]

        user_not_messages = []
        for user in user_not_involved:
            message_count = user_message_map[user]
            user_not_messages.append(message_count)
            
        in_median = median(user_messages)
        no_median = median(user_not_messages)
        delta = in_median - no_median
        hashtag_userMessage_map[hashtag]['delta'] = delta
    
    hashtag_userMessage_map = dict(sorted(hashtag_userMessage_map.items(), key=lambda item:item[1]['delta'], reverse=True))
    first_50_hashtags = dict(islice(hashtag_userMessage_map.items(), 50))
    lead_hashtags = {}
    lead_list = list(first_50_hashtags.keys())
    for hashtag in lead_list:
        lead_hashtags[hashtag] = []
    for index, hashtag in enumerate(list(filtered.keys())):
        if hashtag in lead_list:
            continue
        else:
            similarity = np.zeros(50)
            for index, lead_tag in enumerate(lead_list):
                distance = Levenshtein.distance(hashtag, lead_tag)
                similarity[index] = 1 - (distance / max(len(hashtag), len(lead_tag)))
            max_index = similarity.argmax()
            lead_hashtag = lead_list[max_index]
            lead_hashtags[lead_hashtag].append(hashtag)
    with open(f'{temp_output_dir}/hashtag_group_map.json', 'w') as f:
        json.dump(lead_hashtags, f)
    with open(f'{temp_output_dir}/hashtag_group_map.json', 'r') as f:
        hashtags_group_map = json.load(f)

    openai_client = openai.OpenAI(api_key=openAI_key) if openAI_key else None

    related_hashtags_list = []
    summarize_result = []
    
    for key in hashtags_group_map:
        related_hashtags = key
        for hashtag in hashtags_group_map[key]:
            related_hashtags += ", "
            related_hashtags += hashtag
        related_hashtags_list.append(related_hashtags)
        if openai_client is not None:
            prompt_text = "You are trying to help user summarize several twitter hashtags into a very short sentence (less than 20 words), please do not including instructions in your summarization like(summarizing, summary, etc)"
            message_text = f"First consider the meaning of all hashtags provided, and then summarize them into one short sentence, should be related to most hashtags or the most significant meaning of this hashtag group, consider {instruction}, here are the hashtags: {related_hashtags}\n"
            response = openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": prompt_text},
                    {"role": "user", "content": message_text}
                ]
            )
            summary = response.choices[0].message.content
        else:
            summary = "openai key not provided"
        summary = summary.replace("\r\n", " ").replace("\r", " ").replace("\n", " ")
        summarize_result.append(summary)
    
    ##Start merge hashtags here
    output1_hashtags = {}
    for index, key in enumerate(hashtags_group_map):
        delta = hashtag_userMessage_map[key]['delta']
        summary = summarize_result[index]
        timestamp = hashtag_time_map[key]
        users = hashtag_user_map[key]
        names = [key]
        for follow_hashtag in hashtags_group_map[key]:
            timestamp += hashtag_time_map[follow_hashtag]
            users += hashtag_user_map[follow_hashtag]
            names.append(follow_hashtag)
        
        timestamp = list(set(timestamp))[:3]
        users = list(set(users))
        names = list(set(names))

        output1_title = unidecode(key)
        for index, name in enumerate(names):
            if index == 0:
                continue
            else:
                output1_title += "___"
                output1_title += unidecode(name)
            if index > 2:
                break

        output1_hashtags[output1_title] = {}
        output1_hashtags[output1_title]['window_timestamp'] = timestamp
        output1_hashtags[output1_title]['usersInvolved'] = users
        output1_hashtags[output1_title]['summarization'] = summary
        output1_hashtags[output1_title]['delta'] = delta
    
    hashtag_summary_map = []
    for index, key in enumerate(list(output1_hashtags.keys())):
        hashtagSummary = {}
        hashtagSummary['hashtag_pair'] = key
        hashtagSummary['summarization'] = summarize_result[index]
        hashtagSummary['related_hashtag_group'] = related_hashtags_list[index]
        hashtag_summary_map.append(hashtagSummary)

    df = pd.DataFrame(hashtag_summary_map)

    df.to_csv(f'{folderName}/hashtag_summary_map.csv', index=False, sep=',')
    # df.to_excel(f'{folderName}/hashtag_summary_map.xlsx', index=False)
    with open(f'{folderName}/hashtag_summary_map.txt', 'w') as f:
        for row in hashtag_summary_map:
            key = row['hashtag_pair']
            item = row['summarization']
            f.write(f"{key}: {item}\n")

    with open(f'{folderName}/hashtag_summary_map.json', 'w') as f:
        json.dump(hashtag_summary_map, f)

    with open(f'{temp_output_dir}/output1_hashtags.json', 'w') as f:
        json.dump(output1_hashtags, f)
    return True
    
def addHashtag_WithoutDelta(input_filepath, openAI_key, hashtag_count_threshold=2000, hashtag_lower_limit=50, hashtag_upper_limit=200, instruction = "US China conflict on South China Sea and Philipines stands", folderName = f'{temp_output_dir}'):
    df = pd.read_pickle(input_filepath)
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
    hashtag_user_map_dict = {k: dict(Counter(v).most_common()) for k, v in hashtag_user_map.items()}
    hashtag_freq = sorted(-sum(v.values()) for v in hashtag_user_map_dict.values())
    freq_group = defaultdict(list)
    for k, v in hashtag_user_map_dict.items():
        freq_group[-sum(v.values())].append(k)
    freq_grp_k, freq_grp_v = zip(*sorted(freq_group.items(), key=lambda x: x[0]))
    freq_grp_v_cnt = [len(v) for v in freq_grp_v]
    if len(hashtag_freq) < hashtag_lower_limit:
        hashtag_count_threshold = 0
        filtered = {k: set(v.keys()) for k, v in hashtag_user_map_dict.items() if sum(v.values()) > hashtag_count_threshold}
        print(f"Warning: Threshold adjust to {hashtag_count_threshold} for hashtag selection, {len(hashtag_freq)} hashtags selected, quality is not guaranteed")
    else:
        filtered_count = bisect_left(hashtag_freq, -hashtag_count_threshold)
        if filtered_count < hashtag_lower_limit:
            hashtag_count_threshold = -hashtag_freq[hashtag_lower_limit]
        elif filtered_count > hashtag_upper_limit:
            hashtag_count_threshold = -hashtag_freq[hashtag_upper_limit]
        split_pt = bisect_left(freq_grp_k, -hashtag_count_threshold)
        if sum(freq_grp_v_cnt[:split_pt]) < hashtag_lower_limit:
            selected_hashtags = set()
            for v in freq_grp_v[:split_pt]:
                selected_hashtags.update(v)
            random.shuffle(freq_grp_v[split_pt])
            selected_hashtags.update(freq_grp_v[split_pt][:hashtag_lower_limit - sum(freq_grp_v_cnt[:split_pt])])
            filtered = {k: set(v.keys()) for k, v in hashtag_user_map_dict.items() if k in selected_hashtags}
        elif sum(freq_grp_v_cnt[:split_pt]) > hashtag_upper_limit:
            selected_hashtags = set()
            for v in freq_grp_v[:split_pt - 1]:
                selected_hashtags.update(v)
            random.shuffle(freq_grp_v[split_pt - 1])
            selected_hashtags.update(freq_grp_v[split_pt - 1][:hashtag_upper_limit - sum(freq_grp_v_cnt[:split_pt - 1])])
            filtered = {k: set(v.keys()) for k, v in hashtag_user_map_dict.items() if k in selected_hashtags}
        else:
            filtered = {k: set(v.keys()) for k, v in hashtag_user_map_dict.items() if sum(v.values()) > hashtag_count_threshold}

    print(f"Effective threshold is {hashtag_count_threshold}, {len(filtered.keys())} hashtags selected.")

    hashtag_userMessage_map = {}
    user_involved_total = []

    for hashtag in filtered:
        users_involved = hashtag_user_map[hashtag]
        user_involved_total += users_involved
        hashtag_userMessage_map[hashtag] = {}
        hashtag_userMessage_map[hashtag]['users'] = users_involved

    user_involved_total = list(set(user_involved_total))

    hashtag_userMessage_map = dict(sorted(hashtag_userMessage_map.items(), key=lambda item: len(item[1]['users']), reverse=True))
    first_50_hashtags = dict(islice(hashtag_userMessage_map.items(), 50))
    lead_hashtags = {}
    lead_list = list(first_50_hashtags.keys())
    for hashtag in lead_list:
        lead_hashtags[hashtag] = []
    for index, hashtag in enumerate(list(filtered.keys())):
        if hashtag in lead_list:
            continue
        else:
            similarity = np.zeros(50)
            for index, lead_tag in enumerate(lead_list):
                distance = Levenshtein.distance(hashtag, lead_tag)
                similarity[index] = 1 - (distance / max(len(hashtag), len(lead_tag)))
            max_index = similarity.argmax()
            lead_hashtag = lead_list[max_index]
            lead_hashtags[lead_hashtag].append(hashtag)
    with open(f'{temp_output_dir}/hashtag_group_map.json', 'w') as f:
        json.dump(lead_hashtags, f)
    with open(f'{temp_output_dir}/hashtag_group_map.json', 'r') as f:
        hashtags_group_map = json.load(f)

    openai_client = openai.OpenAI(api_key=openAI_key) if openAI_key else None

    related_hashtags_list = []
    summarize_result = []
    
    for key in hashtags_group_map:
        related_hashtags = key
        for hashtag in hashtags_group_map[key]:
            related_hashtags += ", "
            related_hashtags += hashtag
        related_hashtags_list.append(related_hashtags)
        if openai_client is not None:
            prompt_text = "You are trying to help user summarize several twitter hashtags into a very short sentence (less than 20 words), please do not including instructions in your summarization like(summarizing, summary, etc)"
            message_text = f"First consider the meaning of all hashtags provided, and then summarize them into one short sentence, should be related to most hashtags or the most significant meaning of this hashtag group, consider {instruction}, here are the hashtags: {related_hashtags}\n"
            response = openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": prompt_text},
                    {"role": "user", "content": message_text}
                ]
            )
            summary = response.choices[0].message.content
        else:
            summary = "openai key not provided"
        summary = summary.replace("\r\n", " ").replace("\r", " ").replace("\n", " ")
        summarize_result.append(summary)

    output1_hashtags = {}
    for index, key in enumerate(hashtags_group_map):
        summary = summarize_result[index]
        timestamp = hashtag_time_map[key]
        users = hashtag_user_map[key]
        names = [key]
        for follow_hashtag in hashtags_group_map[key]:
            timestamp += hashtag_time_map[follow_hashtag]
            users += hashtag_user_map[follow_hashtag]
            names.append(follow_hashtag)
        
        timestamp = list(set(timestamp))[:3]
        users = list(set(users))
        names = list(set(names))

        output1_title = unidecode(key)
        for index, name in enumerate(names):
            if index == 0:
                continue
            else:
                output1_title += "___"
                output1_title += unidecode(name)
            if index > 2:
                break

        output1_hashtags[output1_title] = {}
        output1_hashtags[output1_title]['window_timestamp'] = timestamp
        output1_hashtags[output1_title]['usersInvolved'] = users
        output1_hashtags[output1_title]['summarization'] = summary
    
    hashtag_summary_map = []
    for index, key in enumerate(list(output1_hashtags.keys())):
        hashtagSummary = {}
        hashtagSummary['hashtag_pair'] = key
        hashtagSummary['summarization'] = summarize_result[index]
        hashtag_summary_map.append(hashtagSummary)

    with open(f'{folderName}/hashtag_summary_map_without_delta.txt', 'w') as f:
        for row in hashtag_summary_map:
            key = row['hashtag_pair']
            item = row['summarization']
            f.write(f"{key}: {item}\n")
    with open(f'{folderName}/hashtag_summary_map_without_delta.json', 'w') as f:
        json.dump({row['hashtag_pair']: row['summarization'] for row in hashtag_summary_map}, f)

    with open(f'{temp_output_dir}/output1_hashtags_only.json', 'w') as f:
        json.dump(output1_hashtags, f)
    return True

##Merge events and hashtags data to one output1 file save in given location
def mergeEventsHashtags():
    with open(f'{temp_output_dir}/output1_events.json', 'r') as f:
        events = json.load(f)
    
    with open(f'{temp_output_dir}/output1_hashtags.json', 'r') as f:
        hashtags = json.load(f)

    events.update(hashtags)
    with open(f'{temp_output_dir}/output1.json', 'w') as f:
        json.dump(events, f)

##Upload output2 data to backend Port, need Port Path
def upload_Output2(hostPath, description="events_engagement", tag="na", version="1.0"):
    #Get and save output1 data in output2 npy format
    def saveOutput2Data():
        with open (f'{temp_output_dir}/output1.json', 'r') as f:
            output1 = json.load(f)
        total_users = []
        for key in output1:
            users_involved = output1[key]["usersInvolved"]
            total_users += users_involved

        total_users = list(set(total_users))
        user_map = {}
        for index, user in enumerate(total_users):
            user_map[user] = index
        user_engagement = np.zeros((len(total_users), len(list(output1.keys()))))
        event_index = 0
        for key in output1:
            users_involved = output1[key]['usersInvolved']
            for user in users_involved:
                user_index = user_map[user]
                user_engagement[user_index][event_index] = 1
            event_index += 1
        np.save(f'{temp_output_dir}/output2.npy', user_engagement)
    saveOutput2Data()
    #Upload data to corresponding host, with description
    def uploadData():
        config = apollo2_api_client.Configuration(
            host=hostPath,
                api_key={'APIKeyHeader': 'nopass'},
            )
        config.verify_ssl = False
        client = apollo2_api_client.ApiClient(config)
        actor_api = apollo2_api_client.SourceApi(client)
        enrich_api = apollo2_api_client.EnrichmentApi(client)

        dataPath = f'{temp_output_dir}/output2.npy'
        npData = np.load(dataPath)
        file_path_read = f'{temp_output_dir}/output1.json'
        with open(file_path_read, 'r') as f:
            events_data = json.load(f)
        allUsers = []
        for event_key in events_data:
            users_involved = events_data[event_key]['usersInvolved']
            allUsers = allUsers + users_involved

        allUsers = list(set(allUsers))

        ##Create a user map to get the backend userName
        user_idmap = {}
        for batch in tqdm(client.split_list_for_batches(allUsers, 5000)):
            try:
                out = actor_api.source_id_forward_batch_translate(batch)
                user_idmap.update(dict(zip(batch, out)))
            except apollo2_api_client.ApiException:
                for bb in batch:
                    try:
                        out2 = actor_api.source_id_forward_translate(bb)
                        user_idmap[bb] = out2
                    except apollo2_api_client.ApiException:
                        allUsers.remove(bb)
                        pass
        user_dict = {}
        for index, user in enumerate(allUsers):
            user_dict[user] = index

        key_list = list(events_data.keys())
        ##Change all to ascii format
        clean_key_list = [unidecode(string) for string in key_list]
        clean_key_list = [string.replace(" ", "___") for string in clean_key_list]

        ##Create an event map to upload the meta information
        event_map = {}
        for index, event in enumerate(clean_key_list):
            event_map[event] = index

        meta_enrich = apollo2_api_client.ArrayEnrichmentMeta(
            description = description,
            name="event_engagement",
            provider="uiuc-cyphy-event",
            tag=tag,
            version=version,
            data={
                'categories': ["not_engaged", "engaged"]
            },
            label_map= event_map
        )
        enrich_api.enrichments_meta_post(meta_enrich)

        for batch in tqdm(client.split_dict_for_batches(user_idmap, 5000)):
            body = {v: [apollo2_api_client.ArrayEnrichment(
                name="event_engagement",
                provider="uiuc-cyphy-event",
                tag=tag,
                version=version,
                value= npData[user_dict[k]].tolist()
            )] for k, v in batch.items() if k in allUsers}
            if body:
                actor_api.source_enrichments_batch_post(body)
    uploadData()

##Upload output2 hashtag data only to backend Port, need Port Path
def upload_Output_hashtag(hostPath, description="events_engagement", tag="na", version="1.0"):
    #Get and save output1 data in output2 npy format
    def saveOutput2Data():
        with open (f'{temp_output_dir}/output1_hashtags_only.json', 'r') as f:
            output1 = json.load(f)
        total_users = []
        for key in output1:
            users_involved = output1[key]["usersInvolved"]
            total_users += users_involved

        total_users = list(set(total_users))
        user_map = {}
        for index, user in enumerate(total_users):
            user_map[user] = index
        user_engagement = np.zeros((len(total_users), len(list(output1.keys()))))
        event_index = 0
        for key in output1:
            users_involved = output1[key]['usersInvolved']
            for user in users_involved:
                user_index = user_map[user]
                user_engagement[user_index][event_index] = 1
            event_index += 1
        np.save(f'{temp_output_dir}/output2_hashtag.npy', user_engagement)
    saveOutput2Data()
    #Upload data to corresponding host, with description
    def uploadData():
        config = apollo2_api_client.Configuration(
            host=hostPath,
                api_key={'APIKeyHeader': 'nopass'},
            )
        config.verify_ssl = False
        client = apollo2_api_client.ApiClient(config)
        actor_api = apollo2_api_client.SourceApi(client)
        enrich_api = apollo2_api_client.EnrichmentApi(client)

        dataPath = f'{temp_output_dir}/output2_hashtag.npy'
        npData = np.load(dataPath)
        file_path_read = f'{temp_output_dir}/output1_hashtags_only.json'
        with open(file_path_read, 'r') as f:
            events_data = json.load(f)
        allUsers = []
        for event_key in events_data:
            users_involved = events_data[event_key]['usersInvolved']
            allUsers = allUsers + users_involved

        allUsers = list(set(allUsers))

        ##Create a user map to get the backend userName
        user_idmap = {}
        for batch in tqdm(client.split_list_for_batches(allUsers, 5000)):
            try:
                out = actor_api.source_id_forward_batch_translate(batch)
                user_idmap.update(dict(zip(batch, out)))
            except apollo2_api_client.ApiException:
                for bb in batch:
                    try:
                        out2 = actor_api.source_id_forward_translate(bb)
                        user_idmap[bb] = out2
                    except apollo2_api_client.ApiException:
                        allUsers.remove(bb)
                        pass
        user_dict = {}
        for index, user in enumerate(allUsers):
            user_dict[user] = index

        key_list = list(events_data.keys())
        ##Change all to ascii format
        clean_key_list = [unidecode(string) for string in key_list]
        clean_key_list = [string.replace(" ", "___") for string in clean_key_list]

        ##Create an event map to upload the meta information
        event_map = {}
        for index, event in enumerate(clean_key_list):
            event_map[event] = index

        meta_enrich = apollo2_api_client.ArrayEnrichmentMeta(
            description = description,
            name="event_engagement",
            provider="uiuc-cyphy-event",
            tag=tag,
            version=version,
            data={
                'categories': ["not_engaged", "engaged"]
            },
            label_map= event_map
        )
        enrich_api.enrichments_meta_post(meta_enrich)

        for batch in tqdm(client.split_dict_for_batches(user_idmap, 5000)):
            body = {v: [apollo2_api_client.ArrayEnrichment(
                name="event_engagement",
                provider="uiuc-cyphy-event",
                tag=tag,
                version=version,
                value= npData[user_dict[k]].tolist()
            )] for k, v in batch.items() if k in allUsers}
            if body:
                actor_api.source_enrichments_batch_post(body)
    uploadData()