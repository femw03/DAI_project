import argparse
import os
import sys
from typing import List, Literal, Optional, Tuple
import pygame

from .extract_traffic_light_labels import Label, LabelObject, load_labels, find_traffic_light_labels, save_labels_to_file

pygame.init()
pygame.font.init()

RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
WHITE = (255, 255, 255)
WIDTH, HEIGHT = 1280, 720
FONT = pygame.font.Font(None, 36)

class ObjectInfo():
    def __init__(self, label: Label, label_object: LabelObject) -> None:
        self._label = label
        self._label_object = label_object
        self.state : Literal['unknown', 'relevant', 'irrelevant'] = 'unknown'
        self.highlighted = False

    @property
    def bounding_box(self) -> pygame.Rect:
        width = self._label_object.box2d.x2 - self._label_object.box2d.x1
        height = self._label_object.box2d.y2 - self._label_object.box2d.y1

        rect = pygame.Rect((self._label_object.box2d.x1, self._label_object.box2d.y1), (width, height))        
        return rect
    
    @property
    def color(self) -> pygame.Color:
        if self.highlighted:
            return WHITE
        if self.state == 'unknown':
            return RED
        if self.state == 'irrelevant':
            return BLUE
        return GREEN

def draw_bounding_boxes(screen: pygame.Surface, objects: List[ObjectInfo]) -> None:
    for object in objects:
        pygame.draw.rect(
            screen,
            color=object.color,
            rect=object.bounding_box, width=2)
        
def draw_button(screen: pygame.Surface, button_rect = pygame.Rect):
    pygame.draw.rect(screen, (0, 255, 0), button_rect)  # Draw a green button
    text = FONT.render("Click Me", True, (255, 255, 255))  # Button text
    text_rect = text.get_rect(center=button_rect.center)  # Center the text
    screen.blit(text, text_rect)  # Draw the text on the button
        
def find_object_hit(position: Tuple[int, int], objects: List[ObjectInfo]) -> Optional[ObjectInfo]:
    for object in objects:
        if object.bounding_box.collidepoint(position):
            return object
    return None

def within(value, mini, maxi):
    return max(min(value, maxi -1), mini)

def main(label_file: str): 
    if os.path.exists(label_file):
        print(f"Found existing label_file at {os.path.abspath(label_file)}")
        labels =load_labels(label_file, hashed=True)
    else:
        print("No label file was present yet, generating it now")
        labels = find_traffic_light_labels(images_file_path='./bdd100k/images/10k/train', labels_file_path='./bdd100k/labels/det_20/det_train.json')
        save_labels_to_file(labels, label_file)
        

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    pygame.display.set_caption("Traffic light classifier")
    print(100*" ", end='\r')
    image_directory = 'bdd100k/images/10k/train'
    label_index = 0
    label = labels[label_index]
    image = pygame.image.load(f"{image_directory}/{label.name}")
    objects = [ObjectInfo(label, label_object) for label_object in label.labels if label_object.category == 'traffic light']
    objects.sort(key=lambda obj: obj.bounding_box.left)
    button_rect =  pygame.Rect((WIDTH - 100, HEIGHT - 50), (100, 50))
    object_highlight_index = 0
    
    def save_and_go_next_label():
        nonlocal label_index, labels, image, objects, object_highlight_index
        label_index += 1
        label = labels[label_index]
        image = pygame.image.load(f"{image_directory}/{label.name}")
        objects = [ObjectInfo(label, label_object) for label_object in label.labels if label_object.category == 'traffic light']
        objects.sort(key=lambda obj: obj.bounding_box.left)

        object_highlight_index = 0
    
    while True:
        button_visible = all([object.state != 'unknown' for object in objects])
        objects[object_highlight_index].highlighted = True
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                is_left_click = event.button == 1
                mouse_position = pygame.mouse.get_pos()
                object = find_object_hit(mouse_position, objects)
                if object is not None:
                    object.state = 'relevant' if is_left_click else 'irrelevant'
                if button_rect.collidepoint(mouse_position) and button_visible:
                    save_and_go_next_label()
                    
            if event.type == pygame.KEYDOWN:
                key = event.key
                if key == pygame.K_RIGHT:
                    objects[object_highlight_index].highlighted = False
                    object_highlight_index = within(object_highlight_index + 1, 0, len(objects))
                elif key == pygame.K_LEFT:
                    objects[object_highlight_index].highlighted = False
                    object_highlight_index = within(object_highlight_index - 1, 0, len(objects))
                    
                elif key == pygame.K_SPACE:
                    objects[object_highlight_index].state = 'irrelevant'
                    objects[object_highlight_index].highlighted = False
                    object_highlight_index = within(object_highlight_index + 1, 0, len(objects))

                elif key == pygame.K_BACKSPACE:
                    objects[object_highlight_index].state = 'relevant'
                    objects[object_highlight_index].highlighted = False
                    object_highlight_index = within(object_highlight_index + 1, 0, len(objects))
                    
                elif key == pygame.K_RETURN and button_visible:
                    save_and_go_next_label()

                
        screen.blit(image, (0, 0))
        draw_bounding_boxes(screen, objects)
        if button_visible:
            draw_button(screen, button_rect)

        pygame.display.flip()
        clock.tick(60)
        # print(f"Current index: {object_highlight_index}", end='\n')

if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('--label-file', type=str, default='det_traffic_labels.json')
    args = parser.parse_args()
    main(label_file=args.label_file)