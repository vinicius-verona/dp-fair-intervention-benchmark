from Benchmark.dataconf import BenchmarkDatasetConfig


# Test DatsetConfig object creation

def test_dataset_config():

    adult = BenchmarkDatasetConfig(
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
    assert x ==  "BenchmarkDatasetConfig(name=Adult,dir=../data/,target=income,sensitive_attr=sex,categorical_cols=['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country', 'income', 'age', 'hours-per-week'],ordinal_cols=[],continuous_cols=[])"


if __name__ == "__main__":
    test_dataset_config()