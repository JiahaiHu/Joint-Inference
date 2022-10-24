import re
import sys
import time
import datetime
import os
import psutil
import csv

def getTemp():

    if not os.path.exists("temporaryFile"):
        print('No temperature file exists!')
        return
    
    with open("temporaryFile") as fin:
        recordList = []
        for line in fin.readlines():

            core_pattern = 'Core\s[0-9]*:\s*\+[0-9]*\.[0-9]*°C'
            cpu_pattern = 'temp1:\s*\+[0-9]*\.[0-9]*°C'
            float_pattern = '[1-9]\d*(\.)\d*'

            searchObj = re.search(cpu_pattern, line)
            if searchObj:
                recordList.append(re.search(float_pattern, searchObj.group()).group(0).strip())
                continue
                
            searchObj = re.search(core_pattern, line)
            if searchObj:
                recordList.append(re.search(float_pattern, searchObj.group()).group().strip())
                continue
        
        return recordList[-1]

freq_max = 1.5
freq_min = 0.6
freq_unit = 0.3
Temp_limit = 70
Temp_max = Temp_limit * 0.9
Temp_min = Temp_limit * 0.8
status0 = os.popen('sudo cpupower frequency-set -f ' + str(freq_max) + "GHZ")
freq_now = freq_max
print(f'Initial CPU frequency is {freq_max} GHz\n')

index = 0
while True:
    date_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(date_time)
    os.system("sensors > temporaryFile")
    cpuTemp = getTemp()
    cpuUti = psutil.cpu_percent()
    print('CPU temp:', cpuTemp)
    print('CPU uti: %.1f%%' % cpuUti)
    print(f'CPU freq: {freq_now} GHz')
    with open("test.csv", "a", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([index, date_time, cpuTemp, cpuUti])
        index += 1
    if index % 10 == 0:
        if float(cpuTemp) > Temp_max and freq_now > freq_min:
            freq_now = round(freq_now - freq_unit, 2)
            status = os.popen('sudo cpupower frequency-set -f ' + str(freq_now) + "GHZ")
            print(date_time)
            print('CPU temperature is too high!')
            print(f'slow down CPU frequency to {freq_now} GHZ\n')
            with open("freq.txt","w") as f:
                f.write(str(round(freq_now/freq_max, 2)))
            status = os.popen("sudo docker ps | grep 'infer.py' | awk '{print $1}'")
            dockerid = status.read().strip()
            status = os.popen(f'sudo docker cp freq.txt {dockerid}:/home/work/')

        if float(cpuTemp) < Temp_min and freq_now < freq_max:
            freq_now = round(freq_now + freq_unit, 2)
            status = os.popen(f'sudo cpupower frequency-set -f {freq_now} GHZ')
            print(date_time)
            print('CPU temperature is low enough')
            print(f'speed up CPU frequency to {freq_now} GHZ\n')
            with open("freq.txt","w") as f:
                f.write(str(round(freq_now/freq_max, 2)))
            status = os.popen("sudo docker ps | grep 'infer.py' | awk '{print $1}'")
            dockerid = status.read().strip()
            status = os.popen('sudo docker cp freq.txt ' + dockerid + ":/home/work/")
    time.sleep(1)

