from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("yolov8s-cls.pt")   # modelo de clasificación

    model.train(
        data="Autism-4",             # no YAML, es la carpeta
        epochs=200,
        imgsz=224
    )