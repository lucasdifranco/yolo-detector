import numpy as np
import cv2
import time
import json
import os


class imgyolo3:
    def __init__(self) -> None:
        self.labels = []
        (
            self.weight_folder,
            self.weight_file,
            self.cfg_file,
            self.names_file,
        ) = self.load_parameters()

        return None

    def load_parameters(self) -> tuple[str, str, str, str]:
        """
        Function to open json file and get:
            Weight
            Names
            cfg

        Parameters:
            parameters.json (json)

        Returns:
            weight_folder
            weight_file
            cfg_file
            names_file
        """

        print("> Reading json file...")

        with open("parameters.json", "r") as json_file:
            data = json.load(json_file)

        weight_folder = data["folder_name"]

        weight_file = data["weights"][0]
        cfg_file = data["cfg"][0]
        names_file = data["names"][0]

        print("> Done")

        return weight_folder, weight_file, cfg_file, names_file

    def load_network(self) -> tuple[cv2.dnn.Net, list[str], np.ndarray]:
        """
        Function to load network and labels.

        Parameters:
            cfg_path (path),
            weights_path (path),
            names_path (path)

        Return:
            network (Net),
            layers_names_output (list[str]),
            colours (ndarray)
        """

        # Getting paths
        cfg_path = os.path.join(self.weight_folder, self.cfg_file)
        weights_path = os.path.join(self.weight_folder, self.weight_file)
        names_path = os.path.join(self.weight_folder, self.names_file)

        print(f"> Loading network from: {weights_path}")
        # Getting labels
        with open(names_path) as f:
            labels = [line.strip() for line in f]

        # Load the network
        network = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
        network.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        network.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        # Adding colours to each object type
        colours = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")

        # Getting list with names of all layers from YOLO v3 network
        layers_names_all = network.getLayerNames()

        # Returns indexes of layers with unconnected outputs
        layers_indices = network.getUnconnectedOutLayers()
        layers_names_output = [layers_names_all[i - 1] for i in layers_indices]

        print("> Done")

        return network, layers_names_output, colours, labels


class detectionyolo3:
    """
    Detects objects in image based on network.
    """

    def __init__(
        self,
        path,
        network: cv2.dnn.Net,
        probability_minimum: float,
        threshold: float,
        colours: np.ndarray,
        layers_names_output: list[str],
        labels: list[str],
    ):
        self.img_path = path
        self.network = network
        self.probability_minimum = probability_minimum
        self.threshold = threshold
        self.colours = colours
        self.layers_names_output = layers_names_output
        self.labels = labels
        pass
    
    def run_detection(self) -> tuple[np.ndarray, list[str]]:
        """
        Runs image through detection functions and returns image and list with detections and confidences

        Parameters:
            img_path = path
            network = Net
            probability_minimum = float
            threshold = float
            colours = ndarray
            layers_names_output = list[str]
            labels = labels

        Returns:
            image (ndarray)
            detections (list[str])
        """
        self.read_img()
        self.foward_pass()
        self.get_detection()
        self.nm_suppression()
        image, detections = self.out_img()

        return image, detections

    def read_img(self) -> None:
        """
        Funtion to read image, get its shape and blob (Binary Large Object).
        Read more in https://answers.opencv.org/question/50025/what-exactly-is-a-blob-in-opencv/

        Parameters:
            Image (img)
        Return:
            h, w (int,int): height and weight of image
            img_blob (Blob): Blob object with the image data
        """

        # Reading image with opencv (returns a numpy array by default)
        # Note that opencv reads image in BRG format
        self.img_BRG = cv2.imread(self.img_path)
        #self.img_BRG = self.img_path
        # Getting shape of image
        self.h, self.w = self.img_BRG.shape[:2]

        # Getting blob
        # blob = cv2.dnn.blobFromImage(image,
        #  scalefactor=1.0,
        #  size,
        #  mean,
        #  swapRB=True)
        self.blob = cv2.dnn.blobFromImage(
            self.img_BRG, 1 / 255.0, (1024, 1024), swapRB=True, crop=False
        )

    def foward_pass(self) -> None:
        """
        Passes blob "through" the network and returns detection (objects, confidence, boxes)

        Parameters:
            blob (blob)
            network (yolov3 network)

        Returns:
            net_output: detection of the image
        """

        print(f"> Predicting img: {self.img_path}")

        self.network.setInput(self.blob)
        start = time.time()
        self.net_output = self.network.forward(self.layers_names_output)
        end = time.time()
        self.pct_time = end - start

    def get_detection(self) -> None:
        """
        From net_output gets bboxes, confidences and class_numbers
            This process is needed to run cv2.dnn.NMSBoxes (non-maximum suppression of bboxes)
        """
        print("> Getting Predictions...")

        self.bboxes = []
        self.confidences = []
        self.class_number = []

        for result in self.net_output:
            for detected_objects in result:
                scores = detected_objects[5:]
                class_current = np.argmax(scores)
                confidence_current = scores[class_current]

                if confidence_current > self.probability_minimum:
                    box_current = detected_objects[0:4] * np.array(
                        [self.w, self.h, self.w, self.h]
                    )

                    x_center, y_center, box_width, box_height = box_current
                    x_min = int(x_center - (box_width / 2))
                    y_min = int(y_center - (box_height / 2))

                    self.bboxes.append([x_min, y_min, int(box_width), int(box_height)])
                    self.confidences.append(float(confidence_current))
                    self.class_number.append(class_current)

    def nm_suppression(self) -> None:
        """
        Implement Non-maximum suppression of bboxes

        Parameters:
            bboxes (list)
            confidences (list)
            probability_minimum (float)
            threshold (float)
        """
        self.result = cv2.dnn.NMSBoxes(
            self.bboxes, self.confidences, self.probability_minimum, self.threshold
        )

    def out_img(self) -> tuple[any, list[str]]:
        """
        Draws bounding boxes, condifence and labels into image, also checks id theres a object after NMS.

        Parameters:
            results (Sequence[int])

        Returns:
            img_BRG
        """

        out_obj = []

        print("Returning Image...")

        counter = 1

        if len(self.result) > 0:
            for i in self.result.flatten():
                counter += 1

                x_min, y_min = self.bboxes[i][0], self.bboxes[i][1]
                box_width, box_height = self.bboxes[i][2], self.bboxes[i][3]

                x_max = x_min + box_width
                y_max = y_min + box_height

                colour_box_current = self.colours[self.class_number[i]].tolist()

                # Create bounding box on original image

                cv2.rectangle(
                    self.img_BRG,
                    (x_min, y_min),
                    (x_max, y_max),
                    colour_box_current,
                    1,  # You can change thickness in this line
                )

                # Add Label and Confidence on original image
                obj_txt = "{}: {:.4f}".format(
                    self.labels[int(self.class_number[i])], self.confidences[i]
                )

                out_obj.append(obj_txt)

                cv2.putText(
                    self.img_BRG,
                    obj_txt,
                    (x_min, int(y_min - 5)),
                    cv2.FONT_HERSHEY_PLAIN,
                    1.5,
                    colour_box_current,
                    2,
                )

            print("> Prediction took {:.4f} seconds".format(self.pct_time))
            print("> Done")


        return self.img_BRG, out_obj

