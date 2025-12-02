
class Databank:
    def __init__(self):
        pass

    @staticmethod
    def download_from_website(dataset_path: str = "krishnaraj30/salary-prediction-data-simple-linear-regression") -> str:
        import kagglehub

        path = kagglehub.dataset_download(dataset_path)

        if not path:
            print("That path is invalid")
            return None
        
        print(f'The path is valid so here is the path: {path}')

        return path
    
    @staticmethod
    def data_for_logistic_regression(local_path: str = "adult.csv") -> any:
        """Data for logistic regression"""
        import pandas as pd 

        df = pd.read_csv(filepath_or_buffer= local_path)

        return df
data_bank = Databank()

__all__ = ['data_bank']