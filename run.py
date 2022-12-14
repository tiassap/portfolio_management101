#!/usr/bin/env python3
import argparse
import yaml
from utils.general import join
from src.DPG import DPG
from src.marketData_CSV import marketData_CSV

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=False, type=str)
parser.add_argument("--train", action='store_true')

if __name__== "__main__":
	args = parser.parse_args()

	if args.config is not None:
		config_file = open("config/{}.yml".format(args.config))
		config = yaml.load(config_file, Loader=yaml.FullLoader)

		print("Config file: {}".format(config_file))

		# Define dataset
		if config["dataset"]["source"] == "CSV":
			source_data = config["dataset"]["path"]
			currencies = config["dataset"]["currencies"]
			start_period = config["dataset"]["start_period"]
			end_period = config["dataset"]["end_period"]
			dataset = marketData_CSV(csv_filePath=source_data, currencies=currencies, start=start_period, end=end_period)
		
		# Create model class
		model = DPG(config, dataset)

		if args.train:
			model.run_training()

		else:
			pass

	else:
		pass