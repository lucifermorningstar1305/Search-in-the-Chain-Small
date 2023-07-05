import os
import bz2
import argparse
import json
import csv

from rich.progress import track

def process_data(folder_path:str, save_path:str):
    """
    This function is used to process the enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2 
    to obtain the relevant wikipedia passages for the WikiHotpotQA dataset.

    :param file_path: the file path of the original file.
    :param save_path: the path where to save the processed file.

    :returns: None
    """

    count = 0
    with open(save_path, "w", encoding="utf-8-sig") as fp:
        writer = csv.writer(fp, delimiter="\t")

        for root, dirs, files in track(os.walk(folder_path)):
            for file in files:
                file_path = root + "/"+ file

                with bz2.open(file_path, "r") as fb:
                    for line in fb.readlines():
                        data = json.loads(line)
                        row_txt = ""

                        for txt in data["text"]:
                            row_txt += " " + txt
                        writer.writerow([str(count), f"Title: {data['title']} Text: {row_txt}"])
                        count += 1



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--folder_path", "-f", required=True, type=str, help="the location of the wiki-2017001 data")
    parser.add_argument("--save_path", "-s", required=False, type=str, default="./data/hotpotqa/enwiki2017.tsv", help="the path where to save the file.")

    args = parser.parse_args()

    save_folder = "/".join(args.save_path.split("/")[:-1])
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    process_data(args.folder_path, args.save_path)


