
import pygame
def play_sound(file_path ="beep.mp3"):
    pygame.init()
    pygame.mixer.init()

    try:
        sound = pygame.mixer.Sound(file_path)
        sound.play()
        pygame.time.delay(int(sound.get_length() * 50))  # Delay to allow the sound to play
    except pygame.error as e:
        print(f"Error: {e}")
    finally:
        pygame.quit()
