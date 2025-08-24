import polars as pl
from typing import Any

class Model:
    def __init__(self) -> None:
        pass

    def chat(self , prompt:str , system:str|None):
        pass


class Agent:

    """
        how to handle multiple dataset
    """
    def __init__(self , data:pl.DataFrame , predicted:Any = None) -> None:
        self.data = data

    def run(self):
        print("Based on the query, what is the predicted variable")

        print("What variable they want to use ?")

        print("spliting the dataset")

        print("write me a tactics to run the dataset , the work flow of i.e the plan")

        print("writing the code")

        print("double check the code with model")

        print("execute the code")

        print("get the metrics")




def read_csv(filename:str) -> pl.DataFrame:
    data = pl.read_csv(
        filename , 
        try_parse_dates=True , 
        null_values=["" , "NA" , "null"]
    )
    return data

def get_schema(data:pl.DataFrame) -> pl.Schema | None:
    return data.schema


if __name__ == "__main__":
    d = read_csv("./data/train.csv")
    print(get_schema(d))
    prompt = """
        Generate a model to predict the hoursing price
    """
    a = Agent(d, "price")
    a.run()
