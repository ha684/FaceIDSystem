import os
import pygame
import threading
import config

class SoundManager:
    """Class to handle sound notifications for the Face ID Attendance System."""

    def __init__(self):
        """Initialize the sound manager."""
        self.enable_sound = config.ENABLE_SOUND
        
        if self.enable_sound:
            # Initialize pygame mixer for audio
            pygame.mixer.init()
            
            # Create sounds directory if it doesn't exist
            os.makedirs('sounds', exist_ok=True)
            
            # Load sounds
            self.success_sound = self._load_sound('sounds/' + config.SUCCESS_SOUND)
            self.failure_sound = self._load_sound('sounds/' + config.FAILURE_SOUND)
            self.switch_sound = self._load_sound('sounds/' + config.MODE_SWITCH_SOUND)
    
    def _load_sound(self, sound_path):
        """Load a sound file if it exists."""
        if not self.enable_sound:
            return None
        
        try:
            if os.path.exists(sound_path):
                return pygame.mixer.Sound(sound_path)
        except Exception as e:
            print(f"Warning: Could not load sound file {sound_path}: {e}")
        
        return None
    
    def play_success(self):
        """Play the success sound in a non-blocking way."""
        if self.enable_sound and self.success_sound:
            threading.Thread(target=self._play_sound, args=(self.success_sound,)).start()
    
    def play_failure(self):
        """Play the failure sound in a non-blocking way."""
        if self.enable_sound and self.failure_sound:
            threading.Thread(target=self._play_sound, args=(self.failure_sound,)).start()
    
    def play_switch(self):
        """Play the mode switch sound in a non-blocking way."""
        if self.enable_sound and self.switch_sound:
            threading.Thread(target=self._play_sound, args=(self.switch_sound,)).start()
    
    def _play_sound(self, sound):
        """Play a sound in a separate thread."""
        try:
            sound.play()
        except Exception as e:
            print(f"Warning: Could not play sound: {e}")
