from pyspark.sql import SparkSession
#Paths & Constants
DIR = "/Users/rubcuadra/Downloads/Course/"
F_TASK = f"{DIR}/bd_parquets"
S_TASK = f"{DIR}/spb_online"

spark = SparkSession.builder \
	.master("local[*]") \
	.appName("SQLFinalTask") \
	.getOrCreate()

spark.sparkContext.setLogLevel("ERROR")