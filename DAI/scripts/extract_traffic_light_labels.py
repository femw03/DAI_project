import itertools
import json
import os
from typing import Dict, List, Literal, Optional, Union
from dataclasses import dataclass, asdict
from typing import Any
import argparse
import dacite
import dacite.exceptions


@dataclass
class Box2D:
    x1: float
    x2: float
    y1: float
    y2: float


@dataclass
class LabelObject:
    id: str
    attributes: Optional[Dict]
    category: Union[str, Literal["traffic light"]]
    box2d: Box2D


@dataclass
class Label:
    name: str
    labels: List[LabelObject]

    @property
    def contains_traffic_light(self):
        return any(
            [labelObject.category == "traffic light" for labelObject in self.labels]
        )


def deserialize(data: Any, should_log=False) -> Label:
    """
    Recursively convert a dictionary into a dataclass instance.
    """
    try:
        return dacite.from_dict(
            data_class=Label,
            data=data,
            config=dacite.Config(check_types=False, strict=False),
        )
    except dacite.exceptions.MissingValueError as e:
        if should_log:
            print(e)
        return None


def save_labels_to_file(labels: List[Label], file: str) -> None:
    print(f"Writing labels to {os.path.abspath(file)}\r", end="")
    with open(file, "w") as f:
        json.dump([asdict(label) for label in labels], f, indent=4)
        print()
    print(f"Finished writing labels to {os.path.abspath(file)}")


def load_labels(labels_file_path: str, hashed=False) -> List[Label]:
    print(f"Loading labels from {labels_file_path}\r", end="")
    with open(labels_file_path, "r") as f:
        labels_dict = json.load(f)
        if hashed:
            labels_dict = [label_dict for label_dict in labels_dict.values()]
    print(f"finished loading labels from {labels_file_path}, starting deserialization")
    labels: List[Label] = []
    total_labels = len(labels_dict)
    for index, label_dict in enumerate(labels_dict):
        print(f"\rDeserialized {index/total_labels:.2%} of the labels", end="")
        label = deserialize(label_dict)
        if label is not None:
            labels.append(label)
    return labels


def get_image_files(images_file_path: str) -> List[str]:
    return os.listdir(images_file_path)


def find_traffic_light_labels(
    labels_file_path: str, images_file_path: str
) -> List[Label]:
    labels = load_labels(labels_file_path)
    images = get_image_files(images_file_path)
    print(f"amount of labels: {len(labels)}")
    print(f"amount of images: {len(images)}")

    labels_dict = {label.name: label for label in labels}
    labeled_images = [image for image in images if image in labels_dict.keys()]
    valid_labels = [
        labels_dict[key] for key in labeled_images
    ]  # all labels with an image
    print(
        f"{(len(labeled_images)/len(images)):.2%} of the images are present in the labels"
    )

    # Counting the amount of detect objects and traffic lights
    all_objects = list(
        itertools.chain.from_iterable([label.labels for label in valid_labels])
    )
    traffic_lights = [obj for obj in all_objects if obj.category == "traffic light"]
    print(f"{len(all_objects)} objects are detected")
    print(f"{len(traffic_lights)} were traffic lights")

    labels_containing_traffic_light = [
        label for label in valid_labels if label.contains_traffic_light
    ]
    print(
        f"Found {len(labels_containing_traffic_light)} labels containing a traffic light"
    )

    return labels_containing_traffic_light


def main(args: argparse.Namespace):
    labels = find_traffic_light_labels(args.labels, args.images)
    save_labels_to_file(labels, args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--images", type=str, default="./bdd100k/images/10k/train")
    parser.add_argument(
        "--labels", type=str, default="./bdd100k/labels/det_20/det_train.json"
    )
    parser.add_argument("--output", type=str, default="./labels_traffic_lights.json")

    args = parser.parse_args()
    main(args)
