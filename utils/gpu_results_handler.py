import csv

class GpuResultsHandler:
    def __init__(self, filename: str) -> None:

        self.epoch = [] 
        self.gpu_allocated = [] 
        self.gpu_reserved = [] 
        self.gpu_total = [] 
        self.ratio_allocated = [] 

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
                self.gpu_allocated.append(self.processString(row[1]))
                self.gpu_reserved.append(self.processString(row[2]))
                self.gpu_total.append(self.processString(row[3]))
                self.ratio_allocated.append(self.processString(row[4]))