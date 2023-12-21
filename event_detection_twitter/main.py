import argparse
import os
import pathlib

import apollo2_api_client
import event_detection_functions as event_detection
import pandas as pd
import urllib3
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv(pathlib.Path(os.getenv("APP_REPOSITORY_ROOT", ".")).resolve() / ".env")

if __name__ == "__main__":
    openAi_apiKey = os.getenv("OPENAI_API_KEY", None)
    config = apollo2_api_client.Configuration(
        host=os.getenv("APOLLO2_API_ENDPOINT", "https://localhost:8443"),
        api_key={"APIKeyHeader": os.environ["APOLLO2_API_KEY"]},
    )

    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    parser = argparse.ArgumentParser(description="Event Detection in given dataset.")

    parser.add_argument("--identifier", default=None, help="Dataset Identifier")
    parser.add_argument("--platform", default=None, help="Dataset Platform")
    parser.add_argument("--description", default="French Election", help="Dataset description for summarization use")
    parser.add_argument("--upload", action="store_true", help="Upload the files to backend or not")
    parser.add_argument("--hashtag_lower_limit", default=50, help="Lower limit of filtered hashtags")
    parser.add_argument("--hashtag_upper_limit", default=200, help="Upper limit of filtered hashtags")
    parser.add_argument("--threshold_event", default=100, help="Threshold for event user count")
    parser.add_argument("--threshold_hashtag", default=2500, help="Threshold for hashtag count")
    parser.add_argument("--remove_temp", action="store_true", help="Remove the temporary files or not, input T/F")
    parser.add_argument("--artifact_dir", type=str, required=True)
    parser.add_argument("--tag", type=str, default="na")
    parser.add_argument("--version", type=str, default="1.0")

    args = parser.parse_args()

    artifact_dir = pathlib.Path(args.artifact_dir).resolve()
    temp_output_dir = artifact_dir / "temp_output"
    summary_dir = artifact_dir / "summary"

    temp_output_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    input_file = []

    item_get_dict = {}
    if args.platform:
        item_get_dict["platform"] = args.platform
    if args.identifier:
        item_get_dict["identifer"] = args.identifier
    with apollo2_api_client.ApiClient(config) as api_client:
        api_instance = apollo2_api_client.ItemApi(api_client)
        last = -1
        total_msg = api_instance.item_count_get(**item_get_dict)
        print(total_msg)
        with tqdm(total=total_msg, desc="Reading messages", ncols=100) as pbar:
            while True:
                try:
                    api_response = api_instance.item_get(limit=10000, last=last, **item_get_dict)
                    if not api_response:
                        break
                    pbar.update(len(api_response))
                    last = api_response[-1].sid
                    for item in api_response:
                        input_row = {}
                        input_row["name"] = item.author
                        input_row["rawTweet"] = item.data.translated_content_text or item.data.content_text
                        input_row["time_published"] = item.time_published
                        input_file.append(input_row)
                except Exception as e:
                    print("Exception when calling ItemApi->item_get: %s\n" % e)
    input_df = pd.DataFrame(input_file)
    input_df.to_pickle(temp_output_dir / "input_file.pkl")

    input_filepath = temp_output_dir / "input_file.pkl"

    event_detection.updateSavingFolder(temp_output_dir)

    event_detection.preprocess(input_filepath)
    print("Finish Preprocessing")

    event_detection.event_detection()
    print("Finish Event Detection")

    event_detection.postprocess_events(input_filepath)
    print("Finish postprocessing")

    event_detection.merge_events(100)
    print("Finish merging events")

    event_detection.getUserDelta(input_filepath)
    print("Finish ordering through delta")

    event_detection.addEventSummarization(input_filepath, openAi_apiKey, folderName=str(summary_dir))
    print("Finish Events Summary")

    event_detection.addHashtagSummarization(
        input_filepath, openAi_apiKey, hashtag_count_threshold=args.threshold_hashtag, hashtag_lower_limit=args.hashtag_lower_limit, hashtag_upper_limit=args.hashtag_upper_limit, instruction=args.description, folderName=str(summary_dir)
    )
    print("Finish hashtag")

    event_detection.mergeEventsHashtags(str(artifact_dir))
    print("Finish merge outputs")

    if args.upload:
        event_detection.upload_Output2(hostPath=config.host, tag=args.tag, version=args.version)
    if args.remove_temp:
        event_detection.delete_folder_and_contents(f"{temp_output_dir}")
