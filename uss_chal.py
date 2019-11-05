import RPi.GPIO as GPIO
import wiringpi

TRIG = 17
ECHO = 18
SECONDS = 60
start_time, end_time, measure_start_time, count, dist = 0,0,0,0,0

def record_edge_time(channel):
    global start_time, end_time

    # Record rising edge
    if (GPIO.input(ECHO)):
        start_time = wiringpi.micros()
        return
    # Record falling edge
    else:
        end_time = wiringpi.micros()
        return

# GPIO setup
GPIO.setmode(GPIO.BCM)
GPIO.setup(TRIG, GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)

# wiringpi setup
if (wiringpi.wiringPiSetup() == -1):
    exit(1)

# register interrupt handler
GPIO.add_event_detect(ECHO, GPIO.BOTH, callback = record_edge_time)

# record prgoram start time
program_start_time = wiringpi.micros()

# record measure start time
measure_start_time = wiringpi.micros()

# trigger
GPIO.output(TRIG, 0)
wiringpi.delayMicroseconds(20)
GPIO.output(TRIG, 1)
wiringpi.delayMicroseconds(10)
GPIO.output(TRIG, 0)

is_timeout = False
end = False

while (not end):
    # wait 30 ms
    wiringpi.delayMicroseconds(30000)
    
    # time over -> break
    if (wiringpi.micros() - program_start_time > SECONDS * 1000000):
        break

    count = count + 1

    # if echo still HIGH
    if (GPIO.input(ECHO)):
        # if it's first time, it's timeout
        if (not is_timeout):
            print count, '\t', round(dist, 2), '\tcm TO'
            is_timeout = True
        # if already timeout, it's not responding
        else:
            print count, '\t', round(dist, 2), '\tcm NR'

        measure_time = wiringpi.micros() - measure_start_time
        # make sure the sampling rate is 50 ms
        if (measure_time < 50000):
            wiringpi.delayMicroseconds(50000-(measure_time))

        measure_start_time = wiringpi.micros()

    else:
        elapsed_time = end_time - measure_start_time
        
        # if it was timeout, not correctly measured 
        if (is_timeout):
            print count, '\t', round(dist, 2), '\tcm NR'
        else:
            # correctly measured, distance value update
            dist = elapsed_time*340/2*100*0.000001
            print count, '\t', round(dist, 2), '\tcm'

        measure_time = wiringpi.micros() - measure_start_time
        if (measure_time < 50000):
            wiringpi.delayMicroseconds(50000-(measure_time)-125)

        is_timeout = False
        # check for time over
        end = True if (wiringpi.micros() - program_start_time > SECONDS * 1000000) else False

        measure_start_time = wiringpi.micros()

        GPIO.output(TRIG, 0)
        wiringpi.delayMicroseconds(20)
        GPIO.output(TRIG, 1)
        wiringpi.delayMicroseconds(10)
        GPIO.output(TRIG, 0)

GPIO.cleanup()
