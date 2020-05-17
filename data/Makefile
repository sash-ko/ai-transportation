RAW_DATA:=tripdata.csv
DATASET:=data.feather

download:
	wget https://s3-us-west-2.amazonaws.com/nyctlc/yellow_tripdata_2015-01-06.csv.gz
	gunzip yellow_tripdata_2015-01-06.csv.gz
	mv yellow_tripdata_2015-01-06.csv $(RAW_DATA)

create-dataset:
	python preprocess.py --data-file=$(RAW_DATA) --geofence-file=nyc_geofence.geojson --output-file=$(DATASET)

clean:
	# rm yellow_*
	rm *.csv
	rm $(DATA)