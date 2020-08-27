from typing import Dict, List, Optional, Tuple, Union

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from .utilities import ifnone


def load_image(image_path: str):
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    return image


class Visualizer:
    def __init__(self, class_names: Union[Dict[int, str], List[str]]) -> None:
        self.c_names = class_names
        self.colors = np.array(
            [[1, 0, 1], [0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 0], [1, 0, 0]],
            dtype=np.float32,
        )

    def load_image(self, path: str):
        return load_image(path)

    def _get_color(self, c, x, max_val):
        ratio = float(x) / max_val * 5
        i = int(np.floor(ratio))
        j = int(np.ceil(ratio))

        ratio = ratio - i
        r = (1 - ratio) * self.colors[i][c] + ratio * self.colors[j][c]

        return int(r * 255)

    def draw_bboxes(
        self,
        img: Union[np.array, str],
        boxes: np.array,
        classes: Optional[Union[np.array, List]] = None,
        scores: Optional[Union[np.array, List]] = None,
        figsize: Tuple[int, int] = None,
        color=None,
    ):

        if isinstance(img, str):
            img = load_image(img)

        # Get the width and height of the image
        width = img.shape[1]
        height = img.shape[0]

        # Create a figure and plot the image
        sz = ifnone(figsize, (15, 15))
        fig, a = plt.subplots(1, 1, figsize=sz)
        a.imshow(img)

        scores = ifnone(scores, np.repeat(1.0, axis=0, repeats=len(boxes)))

        # Plot the bounding boxes and corresponding labels on top of the image
        for i in range(len(boxes)):
            # Get the ith bounding box
            box = boxes[i]
            # Get the (x,y) pixel coordinates of the lower-left and lower-right corners
            x1 = int(box[0])
            y1 = int(box[1])
            x2 = int(box[2])
            y2 = int(box[3])

            rgb = (1, 0, 0)

            if classes is not None:
                cls_id = classes[i]
                cls_conf = scores[i]
                num_classes = len(self.c_names)
                offset = cls_id * 123457 % num_classes
                red = self._get_color(2, offset, num_classes) / 255
                green = self._get_color(1, offset, num_classes) / 255
                blue = self._get_color(0, offset, num_classes) / 255

                # If a color is given then set rgb to the given color instead
                if color is None:
                    rgb = (red, green, blue)
                else:
                    rgb = color

            width_x = x2 - x1
            width_y = y1 - y2

            # Set the postion and size of the bounding box. (x1, y2) is the pixel coordinate of the
            # lower-left corner of the bounding box relative to the size of the image.
            rect = patches.Rectangle(
                (x1, y2), width_x, width_y, linewidth=2, edgecolor=rgb, facecolor="none"
            )

            # Draw the bounding box on top of the image
            a.add_patch(rect)

            # if classes are given plot the classes and the confidences
            if classes is not None:
                # Create a string with the object class name and the corresponding object class probability
                conf_tx = self.c_names[cls_id] + ": {:.1f}".format(cls_conf)
                # Define x and y offsets for the labels
                lxc = (img.shape[1] * 0.266) / 100
                lyc = (img.shape[0] * 1.180) / 100

                # Draw the labels on top of the image
                a.text(
                    x1 + lxc,
                    y1 - lyc,
                    conf_tx,
                    fontsize=24,
                    color="k",
                    bbox=dict(facecolor=rgb, edgecolor=rgb, alpha=0.8),
                )

        plt.show()
