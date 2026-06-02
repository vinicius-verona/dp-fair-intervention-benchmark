import pandas as pd


for synth in ["mst", "aim"]:
    for type in ["train", "test"]:
        for COMBO in range(1, 7):
            path = f"{synth}/BoD-{COMBO}/"

            # Read the CSV files
            X_train = pd.read_csv(f"{path}X_{type}.csv", index_col=0)
            y_train = pd.read_csv(f"{path}y_{type}.csv", index_col=0)

            # Merge the dataframes horizontally (column-wise)
            train = pd.concat([X_train, y_train], axis=1)

            # Save to train.csv
            train.to_csv(f'{path}{type}.csv')

            print(f"Successfully merged files!")
            print(f"X_{type} shape: {X_train.shape}")
            print(f"y_{type} shape: {y_train.shape}")
            print(f"{type} shape: {train.shape}")
            print(f"\nFirst few rows of {type}.csv:")
            print(train.head())

            # Ask user if we should continue and wait until Y is pressed or leave when N is pressed
            while True:
                user_input = input("Do you want to continue to the next combo? (Y/N): ").strip().upper()
                if user_input == 'Y':
                    break
                elif user_input == 'N':
                    print("Exiting the program.")
                    exit()
                else:
                    print("Invalid input. Please enter 'Y' to continue or 'N' to exit.")