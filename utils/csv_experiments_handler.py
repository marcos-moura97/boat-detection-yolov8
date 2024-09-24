import csv
from typing import List, Union

class CsvExperimentHandler:
    def __init__(self, filename: str) -> None:

        self.model_size = [] 
        self.class_levels = [] 
        self.epochs = [] 
        self.img_resolutions = []

        self.get_data(filename)

    @staticmethod
    def processString(data: str) -> Union[int, str]:
        try:
            return int(data)
        except ValueError:
            return data
        
    def length(self) -> int:
        return len(self.model_size)

    def get_data(self, filename: str) -> None:
        with open(filename, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            for i, row in enumerate(reader):
                if i==0:
                    continue
                self.model_size.append(self.processString(row[0]))
                self.class_levels.append(self.processString(row[1]))
                self.epochs.append(self.processString(row[2]))
                self.img_resolutions.append(self.processString(row[3]))

        if not (len(self.model_size) == len(self.class_levels) == len(self.epochs) == len(self.img_resolutions)):
            print("[Warning] One of the factors has a different number of elements than the others")

    def encode_model_size(self) -> List[int]:
        model = self.model_size
        for i in range(len(model)):
            if model[i] == 'n':
                model[i] = 0
            if model[i] == 's':
                model[i] = 1
            if model[i] == 'm':
                model[i] = 2
        return model
    
    def encode_class_level(self) -> List[int]:
        class_level = self.class_levels
        for i in range(len(class_level)):
            if class_level[i] == 'only_ships':
                class_level[i] = 0
            else:
                class_level[i] += 1
        return class_level

    def encode_epochs(self) -> List[int]:
        epochs = self.epochs
        for i in range(len(epochs)):
            if epochs[i] == 100:
                epochs[i] = 0
            if epochs[i] == 150:
                epochs[i] = 1
            if epochs[i] == 200:
                epochs[i] = 2
            if epochs[i] == 250:
                epochs[i] = 3
            if epochs[i] == 300:
                epochs[i] = 4
        return epochs

    def encode_img_res(self) -> List[int]:    
        img_resolutions = self.img_resolutions
        for i in range(len(img_resolutions)):
            if img_resolutions[i] == 128:
                img_resolutions[i] = 0
            if img_resolutions[i] == 256:
                img_resolutions[i] = 1
            if img_resolutions[i] == 512:
                img_resolutions[i] = 2
            if img_resolutions[i] == 640:
                img_resolutions[i] = 3
        return img_resolutions