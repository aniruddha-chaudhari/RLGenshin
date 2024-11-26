import pygetwindow as gw
import pydirectinput
import time

# Focus on Genshin Impact window
game_window = gw.getWindowsWithTitle('Genshin Impact')[0]
game_window.activate()
time.sleep(0.5)  # Wait for the window to be in focus

# Now send input (e.g., press 'ESC' to open the menu)
pydirectinput.press('esc')
