import os,sys,pdb,cPickle
import numpy as np
import mxnet as mx
import mxnet.gluon as gluon
import mxnet.ndarray as nd
from mxnet.gluon import nn
from mxnet import image
from mxnet import autograd
import math, time
from mxnet.gluon.model_zoo import vision
import matplotlib as mpl
import matplotlib.pyplot as plt

class LOSS_EVAL(mx.metric.EvalMetric):
    def __init__(self,name):
        super(LOSS_EVAL,self).__init__(name)
    def update(self, losss, pred=0):
        for loss in losss:
            if isinstance(loss,mx.nd.NDArray):
                loss = loss.asnumpy()
            self.sum_metric += loss.sum()
            self.num_inst += 1
        return

class YOLO_OUTPUT(nn.HybridBlock): #last layer for yolo.
    def __init__(self,numClass, box_per_cell, verbose=False, **kwargs):
        super(YOLO_OUTPUT, self).__init__(**kwargs)
        chNum = box_per_cell * (numClass + 5) #5 for (iou,x,y,w,h)
        with self.name_scope():
            self.conv=nn.Conv2D(chNum,1,1)
        self.verbose = verbose
        return
    def forward(self, x, *args):
        y = self.conv(x)
        if self.verbose:
            print("yolo_output:",x.shape,'->',y.shape)
        return y

class YOLO(object):
    def __init__(self):
        self.ctx = mx.gpu()
        self.validStep = 10
        self.batchSize = 8
        self.dataShape = 256
        self.box_per_cell = 2
        self.rgb_mean = nd.array([123,117,104])
        self.rgb_std = nd.array([58.395,57.12,57.375])
        self.maxEpochNum = 200
        self.dataRoot = 'dataset/'
        self.outRoot = 'tmp/'
        try:
            os.makedirs(self.outRoot)
        except Exception,e:
            print('exception\t',str(e))
        self.net = None
        self.numClass = 0
        self.classNames = []
        self.trainIter, self.validIter = None, None
    def load_dataset(self,dataName):
        if dataName == "toy":
           self.trainIter = image.ImageDetIter(
               batch_size = self.batchSize,
               data_shape = (3, self.dataShape, self.dataShape),
               path_imgrec = os.path.join( os.path.join(self.dataRoot,dataName), 'train.rec'),
               path_imgidx = os.path.join( os.path.join(self.dataRoot,dataName), 'train.idx'),
               shuffle = True,
               mean = True,
               std = True,
               rand_crop = 1,
               min_object_covered = 0.95,
               max_attempts = 200)
           self.validIter = image.ImageDetIter(
               batch_size=self.batchSize,
               data_shape=(3,self.dataShape,self.dataShape),
               path_imgrec=os.path.join( os.path.join(self.dataRoot,dataName), 'val.rec'),
               shuffle=False, mean=True, std=True)
           self.classNames = 'pikachu,dummy'.split(',')
           self.numClass = len(self.classNames)
           return True
        return False
    def load_model(self,weightFile = None):
        bodyNet = vision.get_model('resnet18_v1', pretrained=True).features
        net = nn.HybridSequential()
        for k in range(len(bodyNet)-2):
            net.add(bodyNet[k])
        output = YOLO_OUTPUT(self.numClass, self.box_per_cell,verbose=False)
        output.initialize()
        net.add(output)
        net.collect_params().reset_ctx(self.ctx)
        if weightFile is not None:
            net.load_params(weightFile,ctx=self.ctx)
        self.net = net
        return True
    def format_net_output(self, Y):
        pred = nd.transpose(Y,(0,2,3,1)) #move channel to last dim
        pred = pred.reshape((0,0,0,self.box_per_cell, self.numClass + 5)) # re-arrange last dim to two dim (B,())
        #here you are responsible to define each field of output
        predCls = nd.slice_axis(pred, begin=0, end=self.numClass,axis=-1)
        predObj = nd.slice_axis(pred, begin=self.numClass, end=self.numClass+1,axis=-1)
        predXY = nd.slice_axis(pred, begin=self.numClass+1, end=self.numClass+3,axis=-1)
        predWH = nd.slice_axis(pred,begin=self.numClass+3,end=self.numClass+5,axis=-1)
        XYWH = nd.concat(predXY, predWH, dim=-1)
        return predCls, predObj, XYWH
    def format_groundtruth(self,labels,XYWH): #generate target online with given labels
        B,H,W,boxNum,_  = XYWH.shape
        boxMask = nd.zeros((B,H,W,boxNum,1),ctx=XYWH.context)
        boxCls = nd.ones_like(boxMask, ctx=XYWH.context) * (-1) #-1 to indicated ignored item
        boxObj = nd.zeros((B,H,W,boxNum,1),ctx = XYWH.context)
        boxXYWH = nd.zeros((B,H,W,boxNum,4), ctx = XYWH.context)
        for b in range(B):
            label = labels[b].asnumpy()
            validLabel = label[np.where(label[:,1] > -0.5)[0],:]
            np.random.shuffle(validLabel) #shuffle to add random following
            for l in validLabel:
                cls,x0,y0,x1,y1 = l #stand label format
                w,h = x1 - x0, y1 - y0
                indx,indy = int(x0*W),int(y0*H) #different to paper, here using left-top to determinet cell

                ious = []
                pws, phs = [1,1],[1,1] #!!!!
                #comparsion between anchor and object bbox(resized to last layer)
                #so anchors stand for size estimation of target in last layer?
                for pw, ph in zip(pws,phs):
                    intersect = np.minimum(pw,w*W) * np.minimum(ph,h*H)
                    ious.append( intersect / (pw*ph + w*h - intersect) )
                bestBoxInd = int(np.argmax(ious))
                boxMask[b,indy,indx,bestBoxInd,:] = 1.0 #select the sell to estimate object
                boxCls[b,indy,indx,bestBoxInd,:] = cls #target class id
                boxObj[b,indy,indx,bestBoxInd,:] = 1.0 #target objectness
                tx,ty = x0 * W - indx, y0 * H - indy #xy is offset from cell left-top(not image)
                tw,th = math.sqrt(w),math.sqrt(h) #for loss reasion, here set target to be sqrted
                boxXYWH[b,indy,indx,bestBoxInd,:] = nd.array([tx,ty,tw,th])
        return boxMask, boxCls, boxObj, boxXYWH
    def valid(self):
        loss_sce = gluon.loss.SoftmaxCrossEntropyLoss(from_logits=False)
        loss_l1 = gluon.loss.L1Loss()
        obj_loss = LOSS_EVAL('obj_loss')
        cls_loss = LOSS_EVAL('cls_loss')
        xywh_loss = LOSS_EVAL('xywh_loss')
        
        positive_weight = 5.0
        negative_weight = 0.1
        class_weight = 1.0
        xywh_weight = 5.0

        self.validIter.reset()
        for batchind, batch in enumerate(self.validIter):
            gtX = batch.data[0].as_in_context(self.ctx)
            gtY = batch.label[0].as_in_context(self.ctx)
            prdY = self.net(gtX)
            prdCls, prdObj, prdXYWH = self.format_net_output(prdY)
            boxMask, boxCls, boxObj, boxXYWH = self.format_groundtruth(gtY, prdXYWH)

            lossCls = loss_sce(prdCls, boxCls, boxMask * class_weight)
            boxWeight = nd.where(boxMask > 0, boxMask * positive_weight, boxMask * negative_weight)
            lossObj = loss_l1(prdObj, boxObj, boxWeight)
            lossXYWH = loss_l1(prdXYWH, boxXYWH, boxMask * xywh_weight)
            obj_loss.update(lossObj)
            cls_loss.update(lossCls)
            xywh_loss.update(lossXYWH)
        print('validation: (%s,%f) (%s,%f) (%s,%f)'%(cls_loss.get()[0], cls_loss.get()[1],\
                                                     obj_loss.get()[0], obj_loss.get()[1],\
                                                     xywh_loss.get()[0], xywh_loss.get()[1]))
        return
    def train(self):
        loss_sce = gluon.loss.SoftmaxCrossEntropyLoss(from_logits=False)
        loss_l1 = gluon.loss.L1Loss()
        obj_loss = LOSS_EVAL('obj_loss')
        cls_loss = LOSS_EVAL('cls_loss')
        xywh_loss = LOSS_EVAL('xywh_loss')

        positive_weight = 5.0
        negative_weight = 0.1
        class_weight = 1.0
        xywh_weight = 5.0

        trainer = gluon.Trainer(self.net.collect_params(),"sgd",{"learning_rate":1,"wd":5e-4})
        for epoch in range(self.maxEpochNum):
            self.trainIter.reset()
            tic = time.time()
            for batchind, batch in enumerate(self.trainIter):
                gtY = batch.label[0].as_in_context(self.ctx)
                gtX = batch.data[0].as_in_context(self.ctx)
                with autograd.record():
                    predY = self.net(gtX)
                    predCls, predObj, predXYWH = self.format_net_output(predY)
                    with autograd.pause():
                        boxMask, boxCls, boxObj, boxXYWH = self.format_groundtruth(gtY,predXYWH)
                    loss0 = loss_sce(predCls, boxCls, boxMask * class_weight)
                    boxWeight = nd.where(boxMask > 0, boxMask * positive_weight, boxMask * negative_weight)
                    loss1 = loss_l1(predObj, boxObj, boxWeight)
                    loss2 = loss_l1(predXYWH, boxXYWH, boxMask * xywh_weight)
                    loss = loss0 + loss1 + loss2
                loss.backward()
                trainer.step(self.batchSize)
                cls_loss.update(loss0)
                obj_loss.update(loss1)
                xywh_loss.update(loss2)
            print('epoch %d(%.2fsec): (%s,%f) (%s,%f) (%s,%f)'%(epoch,time.time() - tic,cls_loss.get()[0], cls_loss.get()[1],\
                                                     obj_loss.get()[0], obj_loss.get()[1],\
                                                     xywh_loss.get()[0], xywh_loss.get()[1]))
            if (1 + epoch) % self.validStep == 0:
                self.valid()
                self.net.save_params(os.path.join(self.outRoot,'yolo%.8d.params'%(epoch+1)))
        return

    def cvt_output_for_predict(self,pred): #how to interprete net output according format_groundtruth()
        predCls,predObj, XYWH = self.format_net_output(pred)
        batchSize,height,width,boxNum,_= XYWH.shape
        X,Y,W,H = XYWH.split(num_outputs=4, axis=-1)
        #pdb.set_trace()
        DY = nd.tile(nd.arange(0,height,repeat=width*boxNum, ctx=XYWH.context).reshape((1,height,width,boxNum,1)), (batchSize,1,1,1,1) )
        DX = nd.tile(nd.arange(0,width,repeat=boxNum,ctx=XYWH.context).reshape((1,1,width,boxNum,1)),(batchSize,height,1,1,1))
        X = (X + DX) / width
        Y = (Y + DY) / height
        W = W ** 2
        H = H ** 2

        
        W = nd.clip(W,0,1)
        H = nd.clip(H,0,1)
        X = nd.clip(X,0,1)
        Y = nd.clip(Y,0,1)
        left = X
        top = Y
        right = nd.clip(left + W,0,1)
        bottom = nd.clip(top + H, 0, 1)
        corners = nd.concat(left,top,right,bottom,dim=-1) #nms requiring corner format
        return predCls, predObj, corners

    def prepare_one_image(self,imagepath):
        with open(imagepath,'rb') as f:
            img = image.imdecode(f.read())
        data = image.imresize(img,self.dataShape,self.dataShape)
        data = (data.astype('float32') - self.rgb_mean) / self.rgb_std
        return data.transpose((2,0,1)).expand_dims(axis=0),img

    def predict_one_image(self,X):
        prdY = self.net(X)
        prdCls, prdObj, prdXYXY = self.cvt_output_for_predict(prdY)
        cid = nd.argmax(prdCls, axis=-1, keepdims=True)
        output = nd.concat(cid, prdObj, prdXYXY,dim=-1)
        output = output.reshape((0,-1,6)) #cid, objectness x0,y0,x1,y1
        output = nd.contrib.box_nms(output) #cid may be changed
        return output

    def box2rect(self,box,color, linewidth = 3):
        box = box.asnumpy()
        return plt.Rectangle(
            (box[0],box[1]), box[2] - box[0], box[3] - box[1],
            fill=False, edgecolor=color,linewidth=linewidth)

    def show_result_in_image(self,img, output, thresh = 0.5):
        #pdb.set_trace()
        plt.imshow(img.asnumpy())
        for row in output:
            row = row.asnumpy()
            #pdb.set_trace()
            cid, score = int(row[0]), row[1]
            if cid < 0 or score < thresh:
                continue
           
            color = 'blue'
            box = row[-4:] * np.array([img.shape[1], img.shape[0]] * 2 )
            #print box,cid
            rect = self.box2rect(nd.array(box),color,2)
            plt.gca().add_patch(rect)
            text = self.classNames[cid] #cid is changed by nms!!!
        plt.show()
        return
    def run_one_image(self,imagepath):
        X,img = self.prepare_one_image(imagepath)
        Y = self.predict_one_image(X.as_in_context(self.ctx))
        self.show_result_in_image(img,Y[0], thresh = 0.9)
        return



