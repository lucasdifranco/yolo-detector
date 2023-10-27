import sys, os
sys.path.append(os.path.join(os.getcwd(),'python/'))
import darknet as dn
import pdb

dn.set_gpu(0)
net = dn.load_net("cfg/yolo-thor.cfg", "/home/pjreddie/backup/yolo-thor_final.weights", 0)
meta = dn.load_meta("cfg/thor.data")

r = dn.detect(net, meta, "data/bedroom.jpg")
