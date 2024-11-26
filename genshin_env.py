import gym
from gym import spaces
import numpy as np  # Fixed syntax error by adding 'as'
import cv2
import pyautogui
import pytesseract
import time
import pygetwindow as gw
import pydirectinput  # Add this import
import vgamepad  # New import
import os  # Add this import

class GenshinEnv(gym.Env):
    def __init__(self):
        super(GenshinEnv, self).__init__()

        # Define action space: Removed ESC
        self.action_space = spaces.Discrete(10)  # 0-5: existing actions, 6-9: mouse movements

        # Define observation space: Processed screen capture (e.g., 84x84 grayscale image)
        self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

        # Initialize game state
        self.state = None
        self.done = False
        self.game_window = None
        self.window_activated = False  # Initialize before activating window
        self.activate_game_window()
        self.action_duration = 0.3  # Add configurable action duration
        self.gamepad = vgamepad.VX360Gamepad()  # Initialize vgamepad
        os.makedirs('screenshots', exist_ok=True)  # Ensure screenshots directory exists

        # Load the template image for locating the distance display
        self.distance_template = cv2.imread('icons\disr.png', 0)  # Path to the distance display template
        if self.distance_template is None:
            raise FileNotFoundError("Distance template image not found!")

        # Load the template image for locating the quest marker
        self.quest_marker_template = cv2.imread('icons\image.png', 0)  # Path to the quest marker template
        if self.quest_marker_template is None:
            raise FileNotFoundError("Quest marker template image not found!")

        self.last_distance_time = time.time()  # Initialize last distance calculation time
        self.previous_distance = None       # To store the previous distance

    def activate_game_window(self):
        """Find and activate the Genshin Impact window."""
        if not self.window_activated:
            for window in gw.getAllTitles():
                if "Genshin Impact" in window:
                    self.game_window = gw.getWindowsWithTitle(window)[0]
                    break
            
            if self.game_window:
                self.game_window.restore()  # Fixed typo here
                self.game_window.activate()
                time.sleep(0.5)  # Increased delay to ensure focus
                print("Genshin Impact window activated.")
                self.window_activated = True
            else:
                raise RuntimeError("Genshin Impact window not found!")

    def reset(self):
        """Reset the environment and return the initial observation."""
        self.done = False
        self.activate_game_window()
        # Press ESC once to ensure clean state
        pyautogui.press('esc')
        time.sleep(0.2)
        pyautogui.keyUp('w')  
        # Press 'v' to show quest marker on the map
        pydirectinput.press('v')
        self.state = self._capture_screen()
        return self._process_image(self.state)

    def step(self, action):
        """Execute one step in the environment based on the given action."""
        reward = 0

        current_time = time.time()
        time_since_last = current_time - self.last_distance_time

        # Capture new game state
        new_state = self._capture_screen()
        processed_state = self._process_image(new_state)

        # Perform the action
        self._perform_action(action)

        # Calculate distance every ~7 seconds
        if time_since_last >= 7:
            quest_distance = self._extract_distance_from_screen()
            if quest_distance is not None:
                if self.previous_distance is not None:
                    distance_change = self.previous_distance - quest_distance
                    if distance_change > 0:
                        reward += distance_change * 0.1  # Reward for decreasing distance
                    else:
                        reward += distance_change * 0.2  # Penalty for increasing distance
                else:
                    reward += 1 / (quest_distance + 1)  # Initial reward

                self.previous_distance = quest_distance
                print(f"Distance to quest marker: {quest_distance}")
            else:
                print("Distance not found.")
                reward -= 1  # Penalty if distance not found

            self.last_distance_time = current_time  # Reset the timer

        # Check if the task is complete
        if self._check_task_complete():
            self.done = True
            reward += 50  # Bonus for task completion

        print(f"Reward: {reward}")  # Print the reward

        return processed_state, reward, self.done, {}

    def _capture_screen(self):
        """Capture the current game screen."""
        if not self.game_window:
            self.activate_game_window()
        
        x, y, width, height = (self.game_window.left, self.game_window.top,
                             self.game_window.width, self.game_window.height)
        screenshot = pyautogui.screenshot(region=(x, y, width, height))
        return np.array(screenshot)

    def _process_image(self, image):
        """Preprocess the image for RL."""
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(grayscale, (84, 84))  # Resize to 84x84
        return np.expand_dims(resized, axis=-1)

    def _perform_action(self, action):
        """Map agent's action to game controls."""
        self.activate_game_window()
        time.sleep(0.2)  # Additional delay

        # Ensure all keys are released before starting
        for key in ['w', 's', 'a', 'd']:
            pydirectinput.keyUp(key)
        time.sleep(0.1)

        if action == 0:  # Forward
            print("Pressing key: w")
            pydirectinput.keyDown('w')
            time.sleep(self.action_duration)
            pydirectinput.keyUp('w')
        elif action == 1:  # Backward
            print("Pressing key: s")
            pydirectinput.keyDown('s')
            time.sleep(self.action_duration)
            pydirectinput.keyUp('s')
        elif action == 2:  # Left
            print("Executing Action: Press a (Left)")
            pydirectinput.keyDown('a')
            time.sleep(self.action_duration)
            pydirectinput.keyUp('a')
        elif action == 3:  # Right
            print("Executing Action: Press d (Right)")
            pydirectinput.keyDown('d')
            time.sleep(self.action_duration)
            pydirectinput.keyUp('d')
        elif action == 4:  # Attack
            print("Executing Action: Click left mouse button (Attack)")
            pydirectinput.click(button='left')
            time.sleep(0.2)
        elif action == 5:  # Interact
            print("Executing Action: Press f (Interact)")
            pydirectinput.press('f')
            time.sleep(0.2)
        elif action == 6:  # Camera move left
            print("Executing Action: Move camera left")
            self.gamepad.left_joystick(x_value=-32768, y_value=0)
            self.gamepad.update()
            time.sleep(0.2)  # Increased duration
            self.gamepad.left_joystick(x_value=0, y_value=0)
            self.gamepad.update()
        elif action == 7:  # Camera move right
            print("Executing Action: Move camera right")
            self.gamepad.left_joystick(x_value=32767, y_value=0)
            self.gamepad.update()
            time.sleep(0.2)  # Increased duration
            self.gamepad.left_joystick(x_value=0, y_value=0)
            self.gamepad.update()
        elif action == 8:  # Camera move up
            print("Executing Action: Move camera up")
            self.gamepad.left_joystick(x_value=0, y_value=32767)
            self.gamepad.update()
            time.sleep(0.2)  # Increased duration
            self.gamepad.left_joystick(x_value=0, y_value=0)
            self.gamepad.update()
        elif action == 9:  # Camera move down
            print("Executing Action: Move camera down")
            self.gamepad.left_joystick(x_value=0, y_value=-32768)
            self.gamepad.update()
            time.sleep(0.2)  # Increased duration
            self.gamepad.left_joystick(x_value=0, y_value=0)
            self.gamepad.update()

        # Ensure all keys are released
        for key in ['w', 's', 'a', 'd']:
            pydirectinput.keyUp(key)

        # Add a small delay after action
        time.sleep(0.1)

    def _extract_distance_from_screen(self):
        """Extract the distance to the quest marker using OCR."""
        try:
            # Find the quest marker location using the template
            quest_marker_pos = self._find_quest_marker()
            if quest_marker_pos is None:
                return None

            # Ensure quest_marker_pos contains two integers
            if not (isinstance(quest_marker_pos, tuple) and len(quest_marker_pos) == 2):
                return None

            # Define the region where the distance is displayed relative to the quest marker position
            distance_region = (
                int(quest_marker_pos[0] - 20),  # X-coordinate offset adjusted to the left
                int(quest_marker_pos[1] + 30),  # Y-coordinate offset from template
                140,                             # Width (increased from 80 to 120)
                25                               # Height
            )

            screenshot = pyautogui.screenshot(region=distance_region)
            screenshot.save("screenshots/distance_screenshot.png")  # Save for inspection

            distance_text = pytesseract.image_to_string(
                screenshot, config='--psm 7 -c tessedit_char_whitelist=0123456789'
            ).strip()

            # Extract numerical value from the text
            distance = ''.join(filter(str.isdigit, distance_text))
            if distance:
                print(f"Extracted Distance: {distance}")  # Retained for distance
                return int(distance)
            return None
        except Exception as e:
            print(f"Failed to extract distance: {e}")
            return None

    def _find_quest_marker(self):
        """Find the location of the quest marker using template matching with retries."""
        try:
            max_retries = 3  # Number of retries
            delay_between_retries = 0.5  # Delay in seconds

            for attempt in range(max_retries):
                # Capture the current game screen in grayscale
                screen = self._capture_screen()
                screen_gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
                cv2.imwrite(f"screenshots/screen_attempt_{attempt + 1}.png", screen_gray)  # Save for inspection

                # Perform template matching
                res = cv2.matchTemplate(screen_gray, self.quest_marker_template, cv2.TM_CCOEFF_NORMED)
                threshold = 0.75  # Lowered threshold for better matching
                loc = np.where(res >= threshold)

                if len(loc[0]) > 0:
                    # Get the top-left coordinates of the first match
                    top_left = (int(loc[1][0]), int(loc[0][0]))  # Ensure integers
                    return top_left
                else:
                    time.sleep(delay_between_retries)

            return None  # Template not found after retries
        except Exception as e:
            return None

    def _check_task_complete(self):
        """Check if the task is complete."""
        # TODO: Implement task completion logic
        return False

    # Remove or comment out methods and code related to actions and rewards