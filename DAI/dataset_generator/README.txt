CLASSIFICATION CLASSES

0 = car
1 = pedestrians
2 = traffic lights
3 = traffic signs
4 = bus
5 = motorcycle 
6 = bicycle 
7 = large vehicles (truck, van, ...)
8 = train 
9 = rider


By running the spawn.py.py file, you spawn some actors into the Carla simulator (pedestrians and vehicles).

By running the extract.py.py file, you can drive around in the CARLA simulation (control a car with basic WASD keys
or use the autopolite by pressing the 'P' key). By pressing the 'l' key, you can start capturing images and segmentation data
that is needed for the actual classification.

The generate.py.py script will processe the images and segmentation data to generate bounding boxes for different object classes, 
such as cars, pedestrians, traffic lights, and more. These bounding boxes are saved in several files:
* draw_bounding_box: Stores images with drawn bounding boxes for visual verification.
* custom_labels: Contains YOLO-format label files for each image, specifying bounding box coordinates and class IDs.
