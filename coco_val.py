from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import pylab
import sys
import os


# please edit this every time you try to run test
network_name = 'mobilenet_v2-ssd'


class Logger(object):
    def __init__(self, fileN="Default.log"):
        self.terminal = sys.stdout
        self.log = open(fileN, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


pylab.rcParams['figure.figsize'] = (10.0, 8.0)

annType = ['segm', 'bbox', 'keypoints']
annType = annType[1]      # specify type here
prefix = 'person_keypoints' if annType == 'keypoints' else 'instances'
print('Running demo for *%s* results.' %(annType))

dataDir = 'C:/Users/Siemo/data/coco'
dataType = 'val2014'
annFile = '%s/annotations/%s_%s.json'%(dataDir, prefix, dataType)
cocoGt = COCO(annFile)
print(annFile)

resFile = './json_result/result_%s.json'%(network_name)
# resFile = '%s/result/result.json'%(dataDir)
coco_txt = open('./coco_result/%s_cocoval.txt'%(network_name), 'w')
sys.stdout = Logger('./coco_result/%s_cocoval.txt'%(network_name))

cocoDt = cocoGt.loadRes(resFile)

imgIds = sorted(cocoGt.getImgIds())
imgIds = imgIds[0:1000]

# running evaluation
cocoEval = COCOeval(cocoGt, cocoDt, annType)
cocoEval.params.imgIds = imgIds
cocoEval.evaluate()
cocoEval.accumulate()
print('this is the coco validation of %s'%(network_name))
cocoEval.summarize()
