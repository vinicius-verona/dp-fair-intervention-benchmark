import sys
from pathlib import Path

# Add src/ to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from Benchmark.dataconf import DatasetConfig


# Test DatsetConfig object creation

def test_dataset_config():

    adult = DatasetConfig(
        name="Adult",
        target="income",   
        root_dir="../data/",
        sensitive_attr="sex",
        categorical_cols=['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country', 'income', 'age', 'hours-per-week'],
        usecols = [
            'age', 'native-country', 'education', 'marital-status', 'occupation', 'relationship',
            'hours-per-week', 'workclass', 'race', 'sex', 'income'
        ] 
    )

    x = str(adult)
    assert x ==  "DatasetConfig(name=Adult,dir=../data/,target=income,sensitive_attr=sex,categorical_cols=['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country', 'income', 'age', 'hours-per-week'],ordinal_cols=[],continuous_cols=[])"


if __name__ == "__main__":
    test_dataset_config()