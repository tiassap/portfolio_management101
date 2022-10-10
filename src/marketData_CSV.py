import pandas as pd
import numpy as np
import os
from datetime import datetime


class marketData_CSV():
	"""
	Processing crypto market data from CSV file.
	Access the data by `marketData_CSV.dataset`.
	...
	Attributes
	----------
	csv_filePath : str
		Path name of csv files of market data.
		Strictly need to use these file names: 'close.csv', 'high.csv', 'low.csv', 'open.csv'
	currencies : list
		List of cryptocurrencies to be used
	start : str
		Starting period. Format: 'YYYY-MM-DD'. Example: '2022-01-01'
	end : str
		End period. Format: 'YYYY-MM-DD'. Example: '2022-01-01'
	"""

	def __init__(self, csv_filePath: str, currencies: list, start: str, end: str):
		self.csv_filePath = csv_filePath
		self.currencies = currencies
		self.start = start
		self.end = end

		# Channels of the array
		self.channels = ['open', 'high', 'low']

		self.dataset = self.__process()

	def __process(self):
		"""
		Processing CSV data into array.
		...
		Return
		------
		dataset : Numpy Array
			dataset.shape : (num_channels, num_currencies, time_period)
		"""
		from functools import reduce

		# Create list containing Pandas dataframe.
		dataset = []
		for i, channel in enumerate(self.channels):
			dataset.append(pd.read_csv(os.path.join(
				self.csv_filePath, channel + '.csv'), usecols=['datetime'] + self.currencies))
			dataset[i]['datetime'] = pd.to_datetime(dataset[i]['datetime'])

		# Check the format of the files.
		self.validate_dataset(dataset)

		# Merge the dataframes by 'datetime' to ensure alignment of data.
		dataset = reduce(lambda left, right: pd.merge(
			left, right, on=['datetime'], how='outer'), dataset)

		# Filter the data by 'datetime', time period of data to be used.
		dataset = dataset[(dataset['datetime'] >= self.start)
						  & (dataset['datetime'] <= self.end)]

		# Convert to numpy array and reshaping.
		dataset = dataset.values[:, 1:]
		dataset = np.array(np.split(dataset, len(self.channels), axis=1))
		dataset = np.moveaxis(dataset, 1, -1)

		return dataset

	def validate_dataset(self, dataset):
		"""Validate csv files format. Return error if incorrect"""
		pass


if __name__ == "__main__":
	path = 'data'
	coins = ['ADA', 'ALGO', 'BTC', 'ETH', 'DOGE']
	start="2021-04-01"
	end="2021-12-31"
	data = marketData_CSV(csv_filePath=path, currencies=coins, start=start, end=end)
	print(data.dataset)
