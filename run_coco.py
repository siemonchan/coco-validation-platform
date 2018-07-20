import tensorflow as tf
import cv2
import time
import numpy as np
import json
import os

result_json = []
time_start = time.time()

# please edit this every time you try to run test
network_name = 'mobilenet_v2-ssd'
# create image results path
os.makedirs('./image_result/%s'%network_name)
# validation data path
print("using coco datasets")
# path to your coco datasets
path = 'C:/Users/Siemo/data/coco/val2014'
# result json path
json_path = 'json_result/result_%s.json'%(network_name)


class DetectionTruth:
    def __init__(self, image_id, category_id, bbox, score):
        self.image_id = image_id
        self.category_id = category_id
        self.bbox = bbox
        self.score = score

    def dump_json(self):
        result_json.append(
            {
                "image_id": self.image_id,
                "category_id": self.category_id,
                "bbox": self.bbox,
                "score": self.score
             }
        )

    def print_self(self):
        print("{")
        print("image_id: " + str(self.image_id))
        print("category_id: " + str(self.category_id))
        print("bbox: " + str(self.bbox))
        print("score: " + str(self.score))
        print("}")


# load model
sess = tf.Session()
saver = tf.train.import_meta_graph('model/%s/model.ckpt.meta'%(network_name))
saver.restore(sess, tf.train.latest_checkpoint('model/%s'%(network_name)))
print("successfully load model")

graph = tf.get_default_graph()

# get io information from graph
image_tensor = graph.get_tensor_by_name('image_tensor:0')
num_detections = graph.get_tensor_by_name('num_detections:0')
detection_classes = graph.get_tensor_by_name('detection_classes:0')
detection_scores = graph.get_tensor_by_name('detection_scores:0')
detection_boxes = graph.get_tensor_by_name('detection_boxes:0')


# loop control param
count = 0
stage = 0
step = 200

for filename in os.listdir(path):
    file_path = os.path.join(path, filename)
    img = cv2.imread(file_path)
    h = img.shape[0]
    w = img.shape[1]

    # reshape the img with batch
    image_data = np.expand_dims(img, axis=0).astype(np.uint8)

    #  run ssd
    result = sess.run([num_detections, detection_classes, detection_scores, detection_boxes], feed_dict={image_tensor: image_data})

    # catch output
    num_detections_res = result[0][0]
    detection_classes_res = result[1][0]
    detection_scores_res = result[2][0]
    detection_boxes_res = result[3][0]

    for i in range(int(num_detections_res)):
        x1 = int(detection_boxes_res[i][1] * w)
        y1 = int(detection_boxes_res[i][0] * h)
        x2 = int(detection_boxes_res[i][3] * w)
        y2 = int(detection_boxes_res[i][2] * h)
        width = x2 - x1
        height = y2 - y1
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0))
        cv2.putText(img, str(int(detection_classes_res[i])), (x1, y1), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
        detection_truth = DetectionTruth(int(filename[13: 25]),
                                         int(detection_classes_res[i]),
                                         [round(x1, 1), round(y1, 1), round(width, 1), round(height, 1)],
                                         float(detection_scores_res[i]))
        detection_truth.dump_json()
        # detection_truth.print_self()

    # cv2.imshow("mobilenetv2_ssd", img)
    # cv2.waitKey(0)
    count = count + 1
    if count - stage * step > step:
        stage = stage + 1
        cv2.imwrite('image_result/%s/'%(network_name) + filename, img)
        print("an image has been saved to: ")

    # show percentage
    print(str(round(count*100/40504, 3))+"% done")

try:
    with open(json_path, 'w') as outfile:
        json.dump(result_json, outfile, indent=0)
except IOError:
    print("dump json error")

print("test down, the result is written into %s", json_path)
time_used = time.time() - time_start
print("time used:" + str(time_used))
