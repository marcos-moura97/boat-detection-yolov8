import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional
from numpy import absolute, subtract, asarray

class AnovaAnalysis:
    def __init__(self, data: Dict[str, List[Any]]) -> None:
        self.df: pd.DataFrame = self.load_data(data)
        self.processed_df: pd.DataFrame = pd.DataFrame()
        self.model = None
        self.anova_table = None

        
        plt.rcParams.update({
            'lines.linewidth': 2,
            'font.size': 25,
            #'xtick.labelsize': 14
            })

    def load_data(self, data: Dict[str, List[Any]]) -> pd.DataFrame:
        return pd.DataFrame(data)

    def process_data(self, indices: List[str]) -> None:
        data: Dict[str, pd.Series] = {index: self.df[index] for index in indices}
        self.processed_df = pd.DataFrame(data)
        self.processed_df = sm.add_constant(self.processed_df)

    def fit_model(self, response_var: str, factors: List[str], interactions: List[str] = []) -> None:
        formula_parts = [f"C({factor})" for factor in factors]
        interaction_parts = [f"C({interaction.replace(':', '):C(')})" for interaction in interactions]
        formula = f"{response_var} ~ " + " + ".join(formula_parts + interaction_parts)
        self.model = ols(formula, data=self.processed_df).fit()
        self.anova_table = sm.stats.anova_lm(self.model, typ=2)

    def predict(self, experiments: pd.DataFrame) -> pd.DataFrame:
        return self.model.predict(experiments)

    def display_anova(self) -> pd.DataFrame:
        if self.anova_table is not None:
            return self.anova_table
        else:
            raise ValueError("Model is not fitted yet. Call `fit_model()` first.")
          
    def plot_difference(self, metric: str, fitted_values: list, save_fig: bool = False) -> None:
        #plt.figure(figsize=(12,8))
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
                
        x = range(1, len(fitted_values)+1)
        error = absolute(subtract(asarray(fitted_values), asarray(self.df[f'{metric}'])))
        data = pd.DataFrame({
            'Experiments': x,
            f'{metric} Experiments': self.df[f'{metric}'].values,
            f'{metric} Model': fitted_values,
            'Error': error
        })
        sns.lineplot(data=data[[f'{metric} Experiments', f'{metric} Model']], ax=axes[0])
        axes[0].set_title(f'{metric} values for Experiments and ANOVA OLS Model')
        axes[1].set_ylim([0, 1])
        sns.move_legend(axes[0], "upper right")
        sns.lineplot(data=data[['Error']], ax=axes[1])
        axes[1].set_title(f'Error from model to real data')
        axes[1].set_ylim([0, 0.02])
        sns.move_legend(axes[1], "upper right")
        if save_fig:
            plt.savefig(f'{metric}_error.png')
        else:
            plt.show()

    def p_values_heatmap(self, response_var: str, save_fig:bool = False) -> None:
        if self.anova_table is None:
            print("Can't plot the heatmap, the table is empty")
            return
                
        anova_p_values = pd.DataFrame({
            'Factor': list(self.anova_table['PR(>F)'].to_dict().keys())[:-1],
            'p_value': list(self.anova_table['PR(>F)'].to_dict().values())[:-1]
        })
        
        plt.figure(figsize=(12,8))
        sns.heatmap(anova_p_values.set_index('Factor').T, annot=True, cmap='coolwarm', cbar=True, linewidths=0.5)
        plt.title('Heatmap of ANOVA p-values')
        if save_fig:
            plt.savefig(f'{response_var}_heatmap.png')
        else:
            plt.show()

    def plot_main_effects(self, factors: List[str], response_var: str, legend: Optional[dict] = None, save_fig: bool = False) -> None:
        num_plots = len(factors)
        fig, axes = plt.subplots(1 if num_plots < 4 else 2, num_plots if num_plots < 4 else num_plots//2, figsize=(7 * num_plots, 3*num_plots))
        if legend is not None:
            df = self.apply_legend(legend)
        else:
            df = self.processed_df

        if num_plots == 1:
            sns.pointplot(x=factors[0], y=response_var, data=df, ax=axes)
        else:
            for i, factor in enumerate(factors):                
                if len(axes) == num_plots:
                    ax_ = axes[i]
                else:
                    ax_ = axes[i//2][i%2]
                sns.pointplot(x=factor, y=response_var, data=df, ax=ax_)
        fig.suptitle('Main Effects Plot')
        
        if save_fig:
            plt.savefig(f'{response_var}_main.png')
        else:
            plt.show()

    def plot_interaction_effects(self, base_factor: str, interacting_factors: List[str], response_var: str, legend: Optional[dict] = None, save_fig: bool = False) -> None:
        if legend is not None:
            df = self.apply_legend(legend)
        else:
            df = self.processed_df
        num_plots = len(interacting_factors)
        plt.figure(figsize=(8 * num_plots, 5*num_plots))
        axes = []
        for i in range(num_plots):
            axes.append(plt.subplot2grid(shape=((num_plots//2)+1,((num_plots//2)+1)*2), loc=((i//2), i*2 if (i//2) == 0 else 1), colspan=2))
        
        for i, factor in enumerate(interacting_factors):
            sns.pointplot(x=base_factor, y=response_var, hue=factor, data=df, ax=axes[i], errorbar=None)
            
            box = axes[i].get_position()
            axes[i].set_position([box.x0, box.y0, box.width*0.75, box.height])
            sns.move_legend(axes[i], "upper left", bbox_to_anchor=(1, 1))
            axes[i].set_title(f'Interaction Effect between {base_factor} and {factor}')

        if save_fig:
            plt.savefig(f'{response_var}_fase_factor_{base_factor}.png')
        else:
            plt.show()

    def apply_legend(self, legend: dict) -> pd.DataFrame:
        df = self.processed_df
        df.rename(columns = {'C_':'C'}, inplace = True)

        for factor in legend.keys():
            for index, val in enumerate(legend[factor]):
                df.loc[df[factor] == index, factor] = val
        return df