# FINAL ORCHESTRATOR PIPELINE

from src.load_data import loading_data

from src.preprocessing_03 import prep_pipeline

from src.model_training_04 import model_training


def final_orchestrator(path: str, target_col: str):
    
    df = loading_data(path)

    prep_df, std_scaler, min_max_scaler, norm_scaler = prep_pipeline(df, target_col)

    model, metrics = model_training(prep_df, target_col)

    print("The full orchestrator was successful...")


if __name__ == "__main__":
    final_orchestrator("./data/diabetes.csv", "Outcome")

