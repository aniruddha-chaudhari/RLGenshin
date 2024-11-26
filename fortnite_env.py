
import gym
from gym import spaces
import numpy as np
import cv2
import pyautogui
import pytesseract
import time
import pygetwindow as gw
import pydirectinput  # Add this import
import vgamepad  # New import

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

        # Perform the action towards the quest marker
        quest_position = self._get_quest_position()
        self._perform_action_towards_quest(quest_position)

        # Capture new game state
        new_state = self._capture_screen()
        processed_state = self._process_image(new_state)

        # Reward logic
        quest_distance = self._get_quest_distance()
        if quest_distance is not None:
            reward = 1 / (quest_distance + 1)  # Higher reward for closer distance
        else:
            reward = -1  # Penalty if quest not found

        # Check if the task is complete
        if self._check_task_complete():
            self.done = True
            reward += 50  # Bonus for task completion

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
            self.gamepad.left_joystick(x_value=-32767, y_value=0)
            self.gamepad.update()
            time.sleep(0.1)
            self.gamepad.left_joystick(x_value=0, y_value=0)
            self.gamepad.update()
        elif action == 7:  # Camera move right
            print("Executing Action: Move camera right")
            self.gamepad.left_joystick(x_value=32767, y_value=0)
            self.gamepad.update()
            time.sleep(0.1)
            self.gamepad.left_joystick(x_value=0, y_value=0)
            self.gamepad.update()
        elif action == 8:  # Camera move up
            print("Executing Action: Move camera up")
            self.gamepad.left_joystick(x_value=0, y_value=32767)
            self.gamepad.update()
            time.sleep(0.1)
            self.gamepad.left_joystick(x_value=0, y_value=0)
            self.gamepad.update()
        elif action == 9:  # Camera move down
            print("Executing Action: Move camera down")
            self.gamepad.left_joystick(x_value=0, y_value=-32767)
            self.gamepad.update()
            time.sleep(0.1)
            self.gamepad.left_joystick(x_value=0, y_value=0)
            self.gamepad.update()

        # Ensure all keys are released
        for key in ['w', 's', 'a', 'd']:
            pydirectinput.keyUp(key)

        # Add a small delay after action
        time.sleep(0.1)

    def _calculate_reward(self, quest_position):
        """Calculate rewards based on quest marker visibility."""
        reward = 0

        # Reward for finding the quest marker
        if quest_position:
            reward += 10  # Reward for seeing the quest marker
        else:
            reward -= 5  # Penalty if quest marker not visible

        return reward

    def _get_health(self):
        """Extract health from the game UI using OCR (placeholder)."""
        # Use PyTesseract or image segmentation to get health values
        return 100

    def _get_quest_position(self):
        """Detect the quest marker position using template matching."""
        template = cv2.imread('icons\Icon_Archon_Quest.webp', 0)  # Load the quest marker template
        w, h = template.shape[::-1]

        screen_gray = cv2.cvtColor(self._capture_screen(), cv2.COLOR_BGR2GRAY)
        res = cv2.matchTemplate(screen_gray, template, cv2.TM_CCOEFF_NORMED)
        threshold = 0.8
        loc = np.where(res >= threshold)

        if len(loc[0]) > 0:
            # Get the first match
            top_left = (loc[1][0], loc[0][0])
            center_x = top_left[0] + w // 2
            center_y = top_left[1] + h // 2
            return (center_x, center_y)
        else:
            return None  # Quest marker not found

    def _perform_action_towards_quest(self, quest_position):
        """Perform action to move towards the quest marker."""
        if quest_position:
            screen_width = self.game_window.width
            agent_x = screen_width // 2  # Assuming agent is at center

            if quest_position[0] < agent_x - 10:
                action = 2  # Left
            elif quest_position[0] > agent_x + 10:
                action = 3  # Right
            else:
                action = 4  # Attack or interact as needed
            self._perform_action(action)
        else:
            self._perform_action(np.random.randint(0, self.action_space.n))

    def _get_quest_distance(self):
        """Estimate distance to quest marker (no longer used)."""
        # This method is deprecated
        return 50

    def _check_task_complete(self):
        """Check if the current task is complete."""
        # Use OCR or template matching to detect task completion
        return False


    def _get_quest_distance(self):
        """Estimate distance to quest marker (no longer used)."""
        # This method is deprecated
        return 50

    def _check_task_complete(self):
        """Check if the current task is complete."""
        # Use OCR or template matching to detect task completion
        return False