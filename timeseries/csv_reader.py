import os
import pandas as pd

def read_and_merge_csv_files(directory_path, filenames, start_date='2010-01-01', end_date='2020-12-31'):
    print(f"Reading and merging CSV files: {filenames}")
    # Initialize an empty DataFrame
    data = pd.DataFrame()

    # Iterate through the specified filenames
    for filename in filenames:
        file_path = os.path.join(directory_path, filename) + '.csv'

        if os.path.isfile(file_path) and file_path.endswith('.csv'):
            # Read the CSV file
            file_data = pd.read_csv(file_path)

            # Extract the 'Date' and 'Close' columns
            file_data = file_data[['Date', 'Close']]

            # Rename the 'Close' column to the file's name without the '.csv' extension
            file_data = file_data.rename(columns={'Close': filename})

            # Merge the data into the main DataFrame
            if data.empty:
                data = file_data
            else:
                data = data.merge(file_data, on='Date', how='outer')
        else:
            print(f"File {file_path} not found")

    data = data.fillna(method='bfill').reset_index(drop=True)
    data['Date'] = pd.to_datetime(data['Date'], utc=True)
    data = data.reset_index(drop=True)
    data = data.sort_values(by='Date')

    start_date = pd.Timestamp(start_date, tz='UTC')
    end_date = pd.Timestamp(end_date, tz='UTC')
    data = data.loc[(data['Date'] >= start_date) & (data['Date'] <= end_date)]

    return data.fillna(method='bfill').reset_index(drop=True)

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

directory_path = 'US-Stock-Dataset/Data/StockHistory'
merged_data = read_and_merge_csv_files(directory_path, ["A", "AAPL", "TSLA", "GOOG", "AMZN", "PYPL"])
print(merged_data)
