from typing import List
from utils import CsvResultsHandler, \
    GpuResultsHandler, \
    CsvExperimentHandler, \
    AnovaAnalysis
from typing import List
import pandas as pd
from itertools import product

def Average(lst: List) -> float: 
    return sum(lst) / len(lst) 

def main(verbose: bool = False) -> None:

    experiments = CsvExperimentHandler("experiments.csv")
    
    mAp = []
    precision = []
    recall = []
    for index in range(experiments.length()):
        
        ms = experiments.model_size[index]
        l = experiments.class_levels[index]
        img = experiments.img_resolutions[index]
        ep =experiments.epochs[index]
        
        try:
            data = CsvResultsHandler(f"D:/Marcos/SMAUG/runs/detect/Ships_detection_yolov8_level{l}_{ms}_{img}_{ep}/results.csv")
            data2 = GpuResultsHandler(f"D:/Marcos/SMAUG/runs/detect/Ships_detection_yolov8_level{l}_{ms}_{img}_{ep}/gpu_results.csv")
            
            if verbose:
                print(f"########## Results for YOLOv8{ms} with Dataset level {l}, img size {img} and {ep} epochs ##########")
                
                print(f"mAp: {max(data.metrics_mAP50_b_)} | \
                Precision: {max(data.metrics_precision_b_)} | \
                Recall: {max(data.metrics_recall_b_)} | \
                malloc: {Average(data2.ratio_allocated)}")
            
            mAp.append(max(data.metrics_mAP50_b_))
            precision.append(max(data.metrics_precision_b_))
            recall.append(max(data.metrics_recall_b_))

        except:
            pass
    
    factors = ['A', 'B', 'C_', 'D']
    

    data = {
        'A': experiments.encode_model_size(),
        'B': experiments.encode_class_level(),
        'C_': experiments.encode_epochs(),
        'D': experiments.encode_img_res(),
        'mAP': mAp,
        'Precision': precision,
        'Recall': recall,
    }    

    legend = {
        'A': ['n', 's', 'm'],
        'B': ['1 class', '2 classes', '15 classes', '25 classes', '50 classes'],
        'C': [100, 150, 200, 250, 300],
        'D': [128, 256, 512, 640]
    }

    analysis = AnovaAnalysis(data=data)
    metrics = ['mAP', 'Precision', 'Recall']
    for met in metrics:
        analysis.process_data(indices=['A', 'B', 'C_', 'D', met])

        # Define the response variable and factors
        factors = ['A', 'B', 'C_', 'D']
        interactions = ['A:B', 'A:C_', 'A:D', 'B:C_', 'B:D', 'C_:D']

        # Fit the model
        analysis.fit_model(response_var=met, factors=factors, interactions=interactions)

        # Display ANOVA results
        anova_results = analysis.display_anova()
        print(anova_results)
        
        analysis.p_values_heatmap(met, save_fig=True)
        factors = ['A', 'B', 'C', 'D']
        analysis.plot_main_effects(factors=factors, response_var=met, legend=legend, save_fig=True)

        # Plot interaction effects
        for base in factors:
            interacting_factors = [ele for ele in factors if ele != base]
            analysis.plot_interaction_effects(base_factor=base, interacting_factors=interacting_factors, response_var=met, legend=legend, save_fig=True)

        
        # Predict the mAP according to ANOVA OLS model
        data_ = pd.DataFrame(
            {
                'A': experiments.model_size,
                'B': experiments.class_levels,
                'C_': experiments.epochs,
                'D': experiments.img_resolutions,
            }
        )
        data_[f'predicted_{met}'] = analysis.predict(data_)

        # Plot the error between predicted and real
        analysis.plot_difference(met, data_[f'predicted_{met}'].values, save_fig=True)

        # Printing predicted mAP
        data_[f'experimental_{met}'] = analysis.df[f'{met}']
        print(data_.sort_values(f'predicted_{met}'))

        # Doing the same analysis for all 300 experiments
        exp_levels = {
            'A': [0, 1, 2],
            'B': [0, 1, 2, 3, 4],
            'C': [0, 1, 2, 3, 4],
            'D': [0, 1, 2, 3],
        }

        all_experiments = list(product(exp_levels['A'], exp_levels['B'], exp_levels['C'], exp_levels['D']))

        all_data = {
            'A': [],
            'B': [],
            'C_': [],
            'D': [],
        }
        
        for tup in all_experiments:
            all_data['A'].append(tup[0])
            all_data['B'].append(tup[1])
            all_data['C_'].append(tup[2])
            all_data['D'].append(tup[3])

        all_data = pd.DataFrame(all_data)
        all_data[f'predicted_{met}'] = analysis.predict(all_data)
        print(all_data.sort_values(f'predicted_{met}'))

        # Doing the same analysis for all 300 experiments
        exp_levels = {
            'A': [0, 1, 2],
            'B': [0, 1, 2, 3, 4],
            'C': [0, 1, 2, 3, 4],
            'D': [0, 1, 2, 3],
        }

        all_experiments = list(product(exp_levels['A'], exp_levels['B'], exp_levels['C'], exp_levels['D']))
        
        all_data = {
            'A': [],
            'B': [],
            'C_': [],
            'D': [],
        }
        for tup in all_experiments:
            all_data['A'].append(tup[0])
            all_data['B'].append(tup[1])
            all_data['C_'].append(tup[2])
            all_data['D'].append(tup[3])

        all_data = pd.DataFrame(all_data)
        all_data[f'predicted_{met}'] = analysis.predict(all_data)
        
        print(all_data.sort_values(f'predicted_{met}'))


if __name__ == "__main__":  
    main()