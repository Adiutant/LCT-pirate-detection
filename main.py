import numpy as np
import pandas as pd
from transformers import ViTFeatureExtractor

from indexer import *
from embeddings import *
import operator
from concurrent.futures import ThreadPoolExecutor
import torch
from panns_inference import AudioTagging
from neural import *
from peaks import *
from db_utils import *
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from trained_vit import ViTForImageClassification

model_audio = AudioTagging(checkpoint_path=None, device='cuda' if torch.cuda.is_available() else 'cpu')


def create_test_csv(model, feature_extractor, database):
    """
    Create a CSV file with the results of the test.
    :param model: neural model
    :param database: database connection
    """
    csv_path = "output.csv"

    pirate_video = "test/"
    pirate_files = os.listdir(pirate_video)
    test_csv = pd.DataFrame(columns=["ID_piracy", "segment", "ID_license", "segment.1"])
    if not os.path.exists(csv_path):
        with open(csv_path, 'w') as f:
            f.write("ID_piracy,segment,ID_license,segment\n")

    def process_file(file, test_csv):
        percent_dict = {}
        if file.endswith(".mp4"):
            print(file)
            dict_data = get_video_embeddings(os.path.join(pirate_video, file), model, feature_extractor, model_audio)
            for table_name in database.table_names():
                table = database.open_table(table_name)
                table_filename = table_name.split("_")[1]
                if table_filename == file:
                    print("")
                full_embedding_video = table.search().where(f"filename = '{table_filename}'").limit(100000).to_list()
                full_embedding_video_vec = [x["vector_video"] for x in full_embedding_video]
                full_embedding_audio = table.search().where(f"filename = '{table_filename}'").limit(100000).to_list()
                full_embedding_audio_vec = [x["vector_audio"] for x in full_embedding_audio]
                matrix = cosine_similarity(dict_data["video"],
                                           np.array(full_embedding_video_vec, dtype=np.float32))
                matrix_audio = cosine_similarity(dict_data["audio"], full_embedding_audio_vec)
                matrix = matrix + matrix_audio
                result_peaks_columns = make_plt_columns(matrix, False)
                if result_peaks_columns["interval"] == "":
                    continue
                else:
                    result_peaks_rows = make_plt_rows(matrix, False)
                    if result_peaks_rows["interval"] == "":
                        continue
                    interval1 = result_peaks_columns["interval"]
                    interval2 = result_peaks_rows["interval"]
                    intervals = f"{interval1} {interval2}"
                    score = 10000 if result_peaks_columns["height"] > 1 else result_peaks_columns["width"] + (
                            result_peaks_columns["height"] * 10)
                    percent_dict[table_filename] = {
                        "score": score,
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
            with open(csv_path, 'a') as f:
                f.write(new_row["ID_piracy"][0] + "," + new_row["segment"][0] + "," + new_row["ID_license"][0] + "," +
                        new_row["segment.1"][0] + "\n")
                # new_row.to_csv(f, header=False, index=False)
            # test_csv = pd.concat([test_csv, new_row], ignore_index=True)
            # test_csv.to_csv("output.csv", index=True)

    # with ThreadPoolExecutor(max_workers=8) as executor:
    #     futures = [executor.submit(process_file, file, test_csv) for file in pirate_files]
    #     print(futures)
    rows = []
    for file in pirate_files:
        process_file(file, test_csv)


if "__main__" == __name__:
    # model = NeuralModel("./models/model.weights.h5")
    model = ViTForImageClassification()
    feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')

    model.load_state_dict(torch.load(os.path.join("models", 'vit_weights.pth')))
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    CONFIDENCE_THRESHOLD = 0.05
    database = get_embeddings_for_directory("index/", model, feature_extractor, model_audio)

    create_test_csv(model, feature_extractor, database)
