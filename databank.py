
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

data_bank = Databank()

__all__ = ['data_bank']