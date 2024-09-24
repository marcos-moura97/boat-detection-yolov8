from utils import YoloTrain
from utils import CsvExperimentHandler

def main() -> None:
    experiments = CsvExperimentHandler("experiments.csv")

    print(f"########## Training {experiments.length()} Models ##########")

    for index in range(experiments.length()):
        
        ms = experiments.model_size[index]
        l = experiments.class_levels[index]
        img = experiments.img_resolutions[index]
        ep =experiments.epochs[index]
        
        print(f"########## Training YOLOv8{ms} with Dataset level {l}, img size {img} and {ep} epochs ##########")
        model = YoloTrain(f'yolov8{ms}.pt')
        model.train_model(data=f'D:/Marcos/SMAUG/Datasets/Ships_detection_yolov8_level{l}/data.yaml',\
                        epochs= ep, imgsz=img , device='0', cache=False, batch=-1, \
                            verbose=True,  name=f"Ships_detection_yolov8_level{l}_{ms}_{img}_{ep}")
                    

if __name__ == "__main__":  
    main()