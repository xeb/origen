import pyttsx3
import argparse

class TTSPlayer:
    def __init__(self, voice_id=None, rate=200):
        """Initialize TTS engine with optional voice ID and speaking rate."""
        self.engine = pyttsx3.init()
        
        # Set properties
        self.engine.setProperty('rate', rate)  # Speed of speech
        
        # Set voice if specified
        if voice_id is not None:
            self.engine.setProperty('voice', voice_id)
    
    def list_voices(self):
        """List all available voices."""
        voices = self.engine.getProperty('voices')
        for idx, voice in enumerate(voices):
            print(f"Voice #{idx}")
            print(f" - ID: {voice.id}")
            print(f" - Name: {voice.name}")
            print(f" - Languages: {voice.languages}")
            print(f" - Gender: {voice.gender}")
            print("----------------------")
        return voices
    
    def speak(self, text, wait=True):
        """Speak the given text."""
        if wait:
            self.engine.say(text)
            self.engine.runAndWait()
        else:
            self.engine.startLoop(False)
            self.engine.say(text)
            self.engine.iterate()
            self.engine.endLoop()
    
    def save_to_file(self, text, filename):
        """Save speech to an audio file."""
        self.engine.save_to_file(text, filename)
        self.engine.runAndWait()
    
    def cleanup(self):
        """Clean up resources."""
        self.engine.stop()

def main():
    parser = argparse.ArgumentParser(description='Text-to-Speech using pyttsx3')
    parser.add_argument('text', help='Text to convert to speech')
    parser.add_argument('--list-voices', action='store_true', help='List available voices')
    parser.add_argument('--voice', type=int, help='Voice index to use')
    parser.add_argument('--rate', type=int, default=200, help='Speaking rate (default: 200)')
    parser.add_argument('--save', help='Save to audio file instead of playing')
    args = parser.parse_args()
    
    try:
        player = TTSPlayer(rate=args.rate)
        
        if args.list_voices:
            voices = player.list_voices()
            if args.voice is not None:
                if 0 <= args.voice < len(voices):
                    player = TTSPlayer(voices[args.voice].id, args.rate)
                else:
                    print(f"Error: Voice index {args.voice} is out of range")
                    return
        
        if args.save:
            player.save_to_file(args.text, args.save)
        else:
            player.speak(args.text)
            
    finally:
        player.cleanup()

if __name__ == "__main__":
    main()