import os
import pickle
import warnings

from average_attrs import average_cosine_similarity

warnings.filterwarnings("ignore")

with open('split_imgs.pkl', 'rb') as f:
    split_imgs = pickle.load(f)

for epoch in [20, 50, 100]:
    weights_path = f'assets/frcnn-{epoch}epochs/frcnn-{epoch}epochs.pt'

    average_scores = []

    for key in sorted(os.listdir('dataset')):
        for i in range(len(split_imgs[key]["test"])):
            image_path = f'dataset/{key}/{split_imgs[key]["test"][i]}'
            average_scores.append(average_cosine_similarity(image_path, weights_path))

    print(f'Average cosine similarity for {epoch} epochs: {sum(average_scores)/len(average_scores)}')
