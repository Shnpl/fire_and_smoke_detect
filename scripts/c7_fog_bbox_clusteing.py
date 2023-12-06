import numpy as np
import os
def iou(boxes, anchors):
    """
    Calculate the IOU between boxes and anchors.

    :param boxes: 2-d array, shape(n, 2)
    :param anchors: 2-d array, shape(k, 2)
    :return: 2-d array, shape(n, k)
    """
    # Calculate the intersection,
    # the new dimension are added to construct shape (n, 1) and shape (1, k),
    # so we can get (n, k) shape result by numpy broadcast
    w_min = np.minimum(boxes[:, 0, np.newaxis], anchors[np.newaxis, :, 0])
    h_min = np.minimum(boxes[:, 1, np.newaxis], anchors[np.newaxis, :, 1])
    inter = w_min * h_min
       
    # Calculate the union
    box_area = boxes[:, 0] * boxes[:, 1]
    anchor_area = anchors[:, 0] * anchors[:, 1]
    union = box_area[:, np.newaxis] + anchor_area[np.newaxis]

    return inter / (union - inter)

def fit(boxes,K = 9, max_iter=10000):
        """
        Run K-means cluster on input boxes.

        :param boxes: 2-d array, shape(n, 2), form as (w, h)
        :return: None
        """
        # If the current number of iterations is greater than 0, then reset
        n_iter = 0

        np.random.seed(0)
        n = boxes.shape[0]

        # Initialize K cluster centers (i.e., K anchors)
        anchors_ = boxes[np.random.choice(n, K, replace=True)]

        labels_ = np.zeros((n,))

        while True:
            n_iter += 1

            # If the current number of iterations is greater than max number of iterations , then break
            if n_iter > max_iter:
                break

            ious_ = iou(boxes, anchors_)
            distances = 1 - ious_
            cur_labels = np.argmin(distances, axis=1)

            # If anchors not change any more, then break
            if (cur_labels == labels_).all():
                break

            # Update K anchors
            for i in range(K):
                anchors_[i] = np.mean(boxes[cur_labels == i], axis=0)

            labels_ = cur_labels
        anchor_area = anchors_[:, 0] * anchors_[:, 1]
        # sort the area from small to large
        index = np.argsort(anchor_area)
        anchors_ = anchors_[index]
        return anchors_

def main():
    label_file_dir = 'datasets/fire_and_smoke_detect/Fog/train/labels'
    label_file_list = os.listdir(label_file_dir)
    label_file_list = [os.path.join(label_file_dir, label_file) for label_file in label_file_list]
    boxes = []
    for label_file in label_file_list:
        with open(label_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if line == '\n':
                    continue
                line = line.strip().split()
                box = [float(line[3]), float(line[4])]
                boxes.append(box)
    print(f"box num:{len(boxes)}")
    boxes = np.array(boxes)
    fitted_boxes = fit(boxes)
    fitted_boxes = fitted_boxes*608
    
    print(fitted_boxes)
    print(fitted_boxes.astype(np.int64))
# boxes = np.array()
# kmeans = KMeans(k=9, max_iter=1000)
# kmeans.fit(boxes)
# anchors = kmeans.anchors_
# print(anchors)

if __name__ == '__main__':
    main()