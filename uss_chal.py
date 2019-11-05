import RPI.GPIO as GPIO
import wiringpi

TRIG = 17
ECHO = 18
SECONDS = 60
start_time, end_time, measure_start_time, count, dist = 0,0,0,0,0

def record_edge_time(channel):
    global start_time, end_time

    if (GPIO.input(ECHO)):
        start_time = wiringpi.micros()
        return
    else:
        end_time = wiringpi.micros()
        return

GPIO.setmode(GPIO.BCM)
GPIO.setup(TRIG, GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)

GPIO.add_event_detect(ECHO, GPIO.BOTH, callback = distance)

program_start_time = wiringpi.micros()
end = False

measure_start_time = wiringpi.micros()
GPIO.output(TRIG, 0)
wiringpi.delayMicroseconds(20)
GPIO.output(TRIG, 1)
wiringpi.delayMicroseconds(10)
GPIO.output(TRIG, 0)

is_timeout = False
while (end):
    count = count + 1

    wiringpi.delayMicroseconds(30000)

    if (GPIO.input(ECHO)):
        if (not is_timeout):
            print count, '\t', round(dist, 2), '\tcm TO'
            is_timeout = True
        else:
            print count, '\t', round(dist, 2), '\tcm NR'

        measure_time = wiringpi.micros() - measure_start_time
        if (measure_time < 50000)
            wiringpi.delayMicroseconds(50000-(measure_time)-125)

        measure_start_time = wiringpi.micros()

    else:
        elapsed_time = end_time - measure_start_time

        if (is_timeout):
            print count, '\t', round(dist, 2), '\tcm NR'
        else:
            dist = elapsed_time*340/2*100*0.000001
            print count, '\t', round(dist, 2), '\tcm'

        measure_time = wiringpi.micros() - measure_start_time
        if (measure_time < 50000):
            wiringpi.delayMicroseconds(50000-(measure_time)-125)

        is_timeout = False
        end = True if (wiringpi.micros() - program_start_time > SECONDS * 1000000) else False

        measure_start_time = wiringpi.micros()

        GPIO.output(TRIG, 0)
        wiringpi.delayMicroseconds(20)
        GPIO.output(TRIG, 1)
        wiringpi.delayMicroseconds(10)
        GPIO.output(TRIG, 0)

GPIO.cleanup()
