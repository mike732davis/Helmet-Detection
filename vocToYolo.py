import os
import xml.etree.ElementTree as ET
import cv2

# Define the paths
xml_folder = 'helmetDataset/annotations'  # Folder where XML files are located
image_folder = 'helmetDataset/images'    # Folder where images are located
output_folder = 'helmetDataset/labels'   # Folder to save YOLO annotations
os.makedirs(output_folder, exist_ok=True)

# Define the class mapping (you can add more classes as needed)
class_mapping = {'With Helmet': 0, 'Without Helmet': 1}

def convert_xml_to_yolo(xml_file, image_width, image_height):
    """
    Converts PASCAL VOC XML annotations to YOLO format.
    """
    # Parse the XML file
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Initialize a list for YOLO annotations
    yolo_annotations = []

    # Loop over each object in the XML file
    for obj in root.findall('object'):
        # Extract class name
        class_name = obj.find('name').text

        # Get the class id (0 for With Helmet, 1 for Without Helmet)
        class_id = class_mapping.get(class_name)
        if class_id is None:
            continue  # Skip if the class is not in our mapping

        # Extract bounding box coordinates
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)

        # Convert bounding box to YOLO format (normalized by image width and height)
        x_center = (xmin + xmax) / 2 / image_width
        y_center = (ymin + ymax) / 2 / image_height
        width = (xmax - xmin) / image_width
        height = (ymax - ymin) / image_height

        # Append the annotation in YOLO format
        yolo_annotations.append(f"{class_id} {x_center} {y_center} {width} {height}")

    return yolo_annotations

def process_annotations():
    """
    Processes all XML files, converts annotations to YOLO format, and saves to a .txt file.
    """
    for xml_file in os.listdir(xml_folder):
        if not xml_file.endswith('.xml'):
            continue  # Skip non-XML files

        # Get the full path of the XML file
        xml_path = os.path.join(xml_folder, xml_file)

        # Parse the XML to get image size
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Get the image size from the XML
        image_name = root.find('filename').text
        image_path = os.path.join(image_folder, image_name)
        image = cv2.imread(image_path)
        image_height, image_width, _ = image.shape

        # Convert the annotations to YOLO format
        yolo_annotations = convert_xml_to_yolo(xml_path, image_width, image_height)

        # Save the YOLO annotations to a .txt file (same name as XML file but with .txt extension)
        yolo_txt_path = os.path.join(output_folder, xml_file.replace('.xml', '.txt'))
        with open(yolo_txt_path, 'w') as f:
            f.write('\n'.join(yolo_annotations))

        print(f"Processed {xml_file} and saved annotations to {yolo_txt_path}")

if __name__ == '__main__':
    process_annotations()
