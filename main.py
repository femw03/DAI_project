import itertools
import json
import os
from typing import List

def load_labels(type: str) -> List:
    with open(f'./bdd100k/labels/det_20/det_{type}.json', 'r') as f:
        labels = json.load(f)
    return labels

def get_image_set_length(type: str) -> int:
    return len(os.listdir(f'./bdd100k/images/10k/{type}'))

def get_image_files(type: str) -> List[str]:
    return os.listdir(f'./bdd100k/images/10k/{type}')

print("loading data")
labels = load_labels('train')
images = get_image_files('train')
print(f"amount of training labels: {len(labels)}")
print(f"amount of training images: {len(images)}")

print("hashing labels", end='')
hashed_labels = {label["name"]:label for label in labels}
print("\rfinished hashing")
labeled_images = []
for image in images:
    corresponding_label = hashed_labels.get(image, None)
    if corresponding_label is not None:
        labeled_images.append(image)
print(f'{(len(labeled_images)/get_image_set_length("train")) * 100:.2f}% of the images are present in the labels')
valid_labels = [hashed_labels[key] for key in labeled_images]
all_objects = list(itertools.chain.from_iterable([label["labels"] for label in valid_labels]))
traffic_lights = [obj for obj in all_objects if obj['category'] == "traffic light"]
print(f'{len(all_objects)} objects are detected')
print(f'{len(traffic_lights)} were traffic lights')
labeled_traffic_lights = [traffic_light for traffic_light in traffic_lights if 'trafficLightColor' in traffic_light['attributes'].keys()]
print(f'{len(labeled_traffic_lights)} were labeled with a color => {len(labeled_traffic_lights)/len(traffic_lights):.2%}%')

print('Writing valid labels to file', end='')
labels_containing_traffic_light = [label for label in valid_labels if any([labels['category'] == 'traffic light' for labels in label['labels']])]

hashed_labels_with_traffic_lights = {traffic_light['name']: hashed_labels[traffic_light['name']] for traffic_light in labels_containing_traffic_light}
with open('det_traffic_labels.json', 'w') as f:
    json.dump(hashed_labels_with_traffic_lights, f, indent=1) 
    print(f"\rFinished writing file to {os.path.abspath(f.name)}")
