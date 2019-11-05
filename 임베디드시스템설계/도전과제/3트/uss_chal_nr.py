import RPi.GPIO as GPIO
import wiringpi
from threading import Thread

TRIG = 17
ECHO = 18
SECONDS = 60
start_time, end_time, measure_start_time, count, dist = 0,0,0,0,0

if (wiringpi.wiringPiSetup() == -1):
    exit(1)

def distance(channel):
    global start_time, end_time

    if (GPIO.input(ECHO)):
        start_time = wiringpi.micros()

        thread = Thread(target=checkNR)
        thread.start()

        return
    else:
        end_time = wiringpi.micros()
        return

def checkNR():
    global dist, count, measure_start_time, start_time, end_time

    while True:
        wiringpi.delayMicroseconds(30000)

        count = count + 1

        # ECHO is 1
        if (GPIO.input(ECHO)):
            print count, '\t', round(dist, 2), '\tcm NR'

            measure_time = wiringpi.micros() - measure_start_time
            if (measure_time < 50000):
                wiringpi.delayMicroseconds(50000-(measure_time))

            measure_start_time = wiringpi.micros()
            continue

        # ECHO is 0
        else:
            elapsed_time = end_time - start_time

            if (elapsed_time >= 30000):
                print count, '\t', round(dist, 2), '\tcm TO(', elapsed_time*0.001, ') ms'

            else:
                dist = (elapsed_time)*340/2*100*0.000001
                                                                                             48,1          Top
