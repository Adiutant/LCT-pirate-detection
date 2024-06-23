import numpy as np
import pandas as pd

from indexer import *
from embeddings import *
import operator
from concurrent.futures import ThreadPoolExecutor
import torch
from transformers import ViTModel, ViTFeatureExtractor
from panns_inference import AudioTagging
from neural import *
from peaks import *
from db_utils import *
from sklearn.metrics.pairwise import cosine_similarity

model_audio = AudioTagging(checkpoint_path=None, device='cuda' if torch.cuda.is_available() else 'cpu')


def create_test_csv(model, database):
    """
    Create a CSV file with the results of the test.
    :param model: neural model
    :param database: database connection
    """
    csv_path = "piracy_val.csv"

    pirate_video = "test/"
    pirate_files = os.listdir(pirate_video)
    test_csv = pd.DataFrame(columns=["ID_piracy", "segment", "ID_license", "segment.1"])

    def process_file(file, test_csv):
        percent_dict = {}
        if file.endswith(".mp4"):
            print(file)
            dict_data = get_video_frames(os.path.join(pirate_video, file), model_audio)
            for table_name in database.table_names():
                table = database.open_table(table_name)
                table_filename = table_name.split("_")[1]
                full_embedding_video = table.search().where(f"filename = '{table_filename}'").limit(100000).to_list()
                full_embedding_video_vec = [x["vector_video"] for x in full_embedding_video]
                full_embedding_audio = table.search().where(f"filename = '{table_filename}'").limit(100000).to_list()
                full_embedding_audio_vec = [x["vector_audio"] for x in full_embedding_audio]
                matrix = make_similarity_with_model(model, dict_data["video"],
                                                    np.array(full_embedding_video_vec, dtype=np.float32))
                matrix_audio = cosine_similarity(dict_data["audio"], full_embedding_audio_vec)
                matrix = matrix + matrix_audio
                result_peaks_columns = make_plt_columns(matrix, True)
                if result_peaks_columns["interval"] == "":
                    continue
                else:
                    result_peaks_rows = make_plt_rows(matrix, True)
                    if result_peaks_rows["interval"] == "":
                        continue
                    interval1 = result_peaks_columns["interval"]
                    interval2 = result_peaks_rows["interval"]
                    intervals = f"{interval1} {interval2}"
                    percent_dict[table_filename] = {
                        "score": result_peaks_columns["width"] + result_peaks_columns["height"],
                        "intervals": f"{intervals}"}

            if len(percent_dict.items()) == 0:
                return
            predicted_license_video = max(percent_dict.items(), key=lambda item: item[1]["score"])[0]
            interval1 = percent_dict[predicted_license_video]["intervals"].split(" ")[0]
            interval2 = percent_dict[predicted_license_video]["intervals"].split(" ")[1]
            new_row = pd.DataFrame({
                'ID_piracy': [file],
                'segment': [interval1],
                'ID_license': [predicted_license_video],
                'segment.1': [interval2]
            })
            print(new_row)
            torch.cuda.empty_cache()
            return new_row

    # with ThreadPoolExecutor(max_workers=8) as executor:
    #     futures = [executor.submit(process_file, file, test_csv) for file in pirate_files]
    #     print(futures)
    rows = []
    for file in pirate_files:
        test_csv = pd.concat([test_csv, process_file(file, test_csv)], ignore_index=True)
    test_csv.to_csv("output.csv", index=True)


if "__main__" == __name__:
    model = NeuralModel("models/model.weights.h5", "model_architecture.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    CONFIDENCE_THRESHOLD = 0.05
    database = get_frames_for_directory("index/", model, model_audio)

    create_test_csv(model, database)
