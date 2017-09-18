import mido
from time import sleep

if __name__ == '__main__':
    inport = mido.open_input()

    while 1:
        msg = inport.receive()
        if msg.type != 'clock':
            print msg

