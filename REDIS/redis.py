import csv
import redis

# Connect to Redis server
r = redis.Redis(host='localhost', port=6379, db=0)

# Open the CSV file
with open('REDIS/eventsdata.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for idx, row in enumerate(reader):
        # Use each row's identifier as a key, or create a new key
        key = f"row:{idx}"
        # Store data as a hash
        r.hmset(key, row)

print("Data loaded into Redis")