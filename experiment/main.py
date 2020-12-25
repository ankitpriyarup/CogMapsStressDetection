import asyncio
import time
import sys
import os
import threading
from pynput import keyboard
from pynput.mouse import Button, Controller
# from lib.cortex import Cortex
from datetime import datetime

mouse = Controller()
terminated = False
def on_press(key):
    try:
        k = key.char 
    except:
        k = key.name 
    if (k == '~'):
        terminated = True
        return False
    if (k == 'left'):
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        file1 = open("log/" + sys.argv[1] + ".txt", "a")
        file1.write("Left Pressed At Time = " + current_time + "\n")
        file1.close()
        mouse.click(Button.left, 1)
    if (k == 'right'):
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        file1 = open("log/" + sys.argv[1] + ".txt", "a")
        file1.write("Right Pressed At Time = " + current_time + "\n")
        file1.close()
        mouse.click(Button.right, 1)
    if (k == 'space'):
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        file1 = open("log/" + sys.argv[1] + ".txt", "a")
        file1.write("Space Pressed At Time = " + current_time + "\n")
        file1.close()
        mouse.click(Button.middle, 1)
    if k in ['left', 'right', 'space']:
        print('Key pressed: ' + k)

async def process_emotiv(cortex, subject_id):
    await cortex.get_user_login()
    await cortex.get_cortex_info()
    await cortex.has_access_right()
    await cortex.request_access()
    await cortex.authorize()
    await cortex.get_license_info()
    await cortex.query_headsets()

    if len(cortex.headsets) > 0:
        await cortex.create_session(activate=True, headset_id=cortex.headsets[0])
        await cortex.create_record(title=subject_id)
        await cortex.subscribe(['pow', 'met'])

        os.startfile('test\montrealstresstest\montrealstresstest.iqx')

        while (True):
            await cortex.get_data()
    else:
        print("No device found!")


def init_emotiv(subject_id):
    cortex = Cortex('cortex_creds')
    asyncio.run(process_emotiv(cortex, subject_id))
    cortex.close()


def startListener():
    print("Keyboard listener initiated!")
    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    listener.join()

if __name__ == '__main__':
    if (len(sys.argv) > 1):
        thread2 = threading.Thread(target=startListener, args=())
        thread2.start()
        os.startfile('test\montrealstresstest\montrealstresstest.iqx')
    #     init_emotiv(sys.argv[1])
