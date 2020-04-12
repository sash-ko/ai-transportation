TRAIN_DATA:=raw_train_data.csv
TEST_DATA:=raw_test_data.csv


download:
	wget https://s3-us-west-2.amazonaws.com/nyctlc/yellow_tripdata_2015-01-06.csv.gz
	gunzip yellow_tripdata_2015-01-06.csv.gz
	mv yellow_tripdata_2015-01-06.csv yellow_tripdata.csv


create-train:
	head -n 1 yellow_tripdata.csv > $(TRAIN_DATA)
	grep "^\d,2015-0[1-4]-\d\d " yellow_tripdata.csv >> $(TRAIN_DATA)
	python3 preprocess.py --data-file=$(TRAIN_DATA) --geofence-file=nyc_geofence.geojson --output-file=train_data.feather


create-test:
	head -n 1 yellow_tripdata.csv > $(TEST_DATA)
	grep "^\d,2015-0[5-7]-\d\d " yellow_tripdata.csv >> $(TEST_DATA)
	python3 preprocess.py --data-file=$(TEST_DATA) --geofence-file=nyc_geofence.geojson --output-file=test_data.feather


clean:
	rm yellow_*
	rm *.csv
	rm $(TRAIN_DATA)
	rm $(TEST_DATA)