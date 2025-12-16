import psutil
import time 
import csv 
import datetime 

total_duration = 7200  # 2 hours in seconds
interval =5  # data collection interval in seconds
total_iterations = total_duration // interval

data=[] # to collect the data rows
counter =0 # to count iterations
start_time = time.time()

#create a csv file and write the header
csv_file_path = 'system_performance_data.csv'
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Timestamp', 'CPU_Usage_Percent', 'Memory_Usage_Percent', 'Disk_Usage_Percent'])
    while counter < total_iterations:
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        cpu_usage = psutil.cpu_percent(interval=1)
        memory_usage = psutil.virtual_memory().percent
        disk_usage = psutil.disk_usage('/').percent

        # Append the collected data to the list
        data.append([timestamp, cpu_usage, memory_usage, disk_usage])

        # Write the collected data to the CSV file
        writer.writerow([timestamp, cpu_usage, memory_usage, disk_usage])

        counter += 1
        elapsed_time = time.time() - start_time
        sleep_time = (counter * interval) - elapsed_time
        if sleep_time > 0:
            time.sleep(sleep_time)

print(f"Data collection complete. Data saved to {csv_file_path}")
