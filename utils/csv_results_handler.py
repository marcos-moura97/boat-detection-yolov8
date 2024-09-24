import csv

class CsvResultsHandler:
    def __init__(self, filename: str) -> None:

        self.epoch = [] 
        self.train_box_loss = [] 
        self.train_cls_loss = [] 
        self.train_dfl_loss = [] 
        self.metrics_precision_b_ = [] 
        self.metrics_recall_b_ = []
        self.metrics_mAP50_b_ = []
        self.metrics_mAP50_95_b_ = [] 
        self.val_box_loss = []
        self.val_cls_loss = [] 
        self.val_dfl_loss = [] 
        self.lr_pg0 = [] 
        self.lr_pg1 = [] 
        self.lr_pg2 = []

        self.get_data(filename)

    @staticmethod
    def processString(data: str) -> float:
        return float(data.replace(" ", ""))

    def get_data(self, filename: str) -> None:
        with open(filename, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            for i, row in enumerate(reader):
                if i==0:
                    continue
                self.epoch.append(self.processString(row[0]))
                self.train_box_loss.append(self.processString(row[1]))
                self.train_cls_loss.append(self.processString(row[2]))
                self.train_dfl_loss.append(self.processString(row[3]))
                self.metrics_precision_b_.append(self.processString(row[4]))
                self.metrics_recall_b_.append(self.processString(row[5]))
                self.metrics_mAP50_b_.append(self.processString(row[6]))
                self.metrics_mAP50_95_b_.append(self.processString(row[7]))
                self.val_box_loss.append(self.processString(row[8]))
                self.val_cls_loss.append(self.processString(row[9]))
                self.val_dfl_loss.append(self.processString(row[10]))
                self.lr_pg0.append(self.processString(row[11]))
                self.lr_pg1.append(self.processString(row[12]))
                self.lr_pg2.append(self.processString(row[13]))