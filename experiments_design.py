from utils import ExperimentsDesign

def main() -> None:
    model_size = ['n', 's', 'm']
    class_levels = [0, 1, 2, 3, 'only_ships']
    epochs = [100, 150, 200, 250, 300]
    img_resolutions = [128, 256, 512, 640]

    factors = (model_size, class_levels, epochs, img_resolutions)

    design = ExperimentsDesign(factors)
    df_fractional_factorial = design.to_dataframe()
    print(df_fractional_factorial.head())
    design.save_to_csv("experiments.csv")

if __name__ == '__main__':
    main()
