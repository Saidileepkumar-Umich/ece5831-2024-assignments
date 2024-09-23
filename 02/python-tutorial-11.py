import pandas as pd
import sys

for path in sys.path:
    print(path)

def main():
    # Create a DataFrame
    df = pd.read_csv('cars.csv')
    print(df)

if __name__ == '__main__':
    main()