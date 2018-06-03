import os,sys,pdb
import numpy as np
from yolo import YOLO

clf = YOLO()
clf.load_dataset('toy') #determine class number
clf.load_model("tmp/yolo%.8d.params"%189)
clf.run_one_image('dataset/toy/test.jpg')