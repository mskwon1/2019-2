import RPi.GPIO as GPIO
import wiringpi

TRIG = 17
ECHO = 18
SECONDS = 60
start_time, end_time, measure_start_time, count, dist = 0,0,0,0,0

if (wiringpi.wiringPiSetup() == -1):
    exit(1)

def distance(channel):
    global count, start_time, end_time, dist, measure_start_time
    if (GPIO.input(ECHO)):
        start_time = wiringpi.micros()
        return

    else:
        end_time = wiringpi.micros()

        count = count + 1
        elapsed_time = end_time - start_time

        if (elapsed_time >= 30000):
            print count,'\t', round(dist,2), '\tcm TO(', elapsed_time*0.001, ') ms'

        else:
            dist = (elapsed_time)*340/2*100*0.000001
            print count,'\t', round(dist,2), '\tcm'

        if (GPIO.input(ECHO)):
            print count,'\t', round(dist,2), '\tcm NR'
            return

        measure_time = wiringpi.micros()-measure_start_time
        if (measure_time < 50000-125):
            wiringpi.delayMicroseconds(50000-(measure_time)-125)

        GPIO.output(TRIG, 1)
        wiringpi.delayMicroseconds(10)
        GPIO.output(TRIG, 0)
        measure_start_time = wiringpi.micros()
        return

GPIO.setmode(GPIO.BCM)
GPIO.setup(TRIG, GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)

GPIO.add_event_detect(ECHO, GPIO.BOTH, callback = distance)

GPIO.output(TRIG, 1)
wiringpi.delayMicroseconds(10)
GPIO.output(TRIG, 0)
measure_start_time = wiringpi.micros()