import serial
import time
#import keyboard


ser =serial.Serial(port="COM4",baudrate=115200,bytesize=8,timeout=2,stopbits=serial.STOPBITS_ONE)

while True:
    command=input("Master command: ")
    ser.write(command.encode('Ascii'))
    receive =ser.readline()
    print(receive.decode('Ascii'))
    time.sleep(1)

    #if keyboard.is_pressed('q'):
    #    print("quited")
    #    break

ser.close()

