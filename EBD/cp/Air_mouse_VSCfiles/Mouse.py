# import serial
# import pyautogui
# ser=serial.Serial('com4',9600)
# while 1:
#     k=ser.read(8)
#     cursor=k[:8]
#     # click=k[6:]
#     x=cursor[:4]   
#     y=cursor[4:]
#     # l=click[0]
#     # r=click[1]
#     xcor=int(x.decode('utf-8'))
#     ycor=int(y.decode('utf-8'))
#     # pyautogui.moveTo(xcor,ycor)
#     # if l==49:
#     #     pyautogui.click(clicks=2)
#     # elif r==49:
#     #     pyautogui.click(button='right', clicks=2)
#     print(xcor)
#     print(ycor)
#     pyautogui.moveTo(xcor,ycor)


import serial
import pyautogui

ser = serial.Serial('COM3', 9600)  # Replace 'COM4' with the name of your serial port
while True:
    data = ser.readline().decode().strip()
    # cursor=data[:6]
    # x = cursor[:3]
    # y = cursor[3:]
    print(data)
    if data:
        values = data.split('#')
        if len(values) >= 2:
            try:
                x, y = map(int, values[:2])
                pyautogui.moveTo(x, y)
                print(x,y)
            except ValueError:
                print('Received non-integer data:', data)
        else:
            print(data)

            # pyautogui.moveTo(100,200)



# import serial

# ser = serial.Serial('COM4', 9600)  # Replace 'COM3' with the name of your serial port
# while True:
#     data = ser.readline().decode().strip()
#     if data:
#         x, y = data.split(' ')
#         print('x:', x, 'y:', y)

   

