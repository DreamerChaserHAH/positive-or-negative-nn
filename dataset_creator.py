import random;
import os;
import csv;

for x in range(10000):
    number = random.randint(-5000, 5000)
    is_great_than_0 = (number > 0) + 0.0
    with open("dataset_input.csv", 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow([number])
    with open("dataset_output.csv", 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow([is_great_than_0])
    