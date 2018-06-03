import os,sys,pdb
import numpy as np
from yolo import YOLO

clf = YOLO()
clf.load_dataset('toy')
clf.load_model()
clf.train()
