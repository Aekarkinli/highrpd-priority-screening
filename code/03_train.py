"""Train one detector configuration. Usage: python code/03_train.py <model> <data_yaml> <seed> <run_name> [epochs]"""
import sys, os, time, json
CACHE=os.environ.get("YOLO_CACHE","ram"); BATCH=int(os.environ.get("YOLO_BATCH","32")); WORKERS=int(os.environ.get("YOLO_WORKERS","2"))
from ultralytics import YOLO
def main():
    model,data,seed,name=sys.argv[1],sys.argv[2],int(sys.argv[3]),sys.argv[4]
    epochs=int(sys.argv[5]) if len(sys.argv)>5 else 100
    t=time.time()
    m=YOLO(model)
    m.train(data=data,epochs=epochs,imgsz=640,batch=BATCH,seed=seed,deterministic=True,patience=25,
            project='runs',name=name,exist_ok=True,plots=False,workers=WORKERS,verbose=False,cache=CACHE)
    json.dump(dict(model=model,data=data,seed=seed,epochs=epochs,train_time_s=time.time()-t),open(f'runs/{name}/meta.json','w'))
if __name__=='__main__': main()
