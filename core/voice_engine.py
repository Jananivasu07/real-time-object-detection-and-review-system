import pyttsx3
import threading
import queue

class VoiceEngine:
    """
    Real-time Text-to-Speech engine for object announcements.
    Runs in a separate thread to prevent UI blocking.
    """
    def __init__(self):
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150) # Speed of speech
        self.q = queue.Queue()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
        self.last_spoken = ""

    def _worker(self):
        while not self.stop_event.is_set():
            try:
                text = self.q.get(timeout=1)
                self.engine.say(text)
                self.engine.runAndWait()
            except queue.Empty:
                continue

    def announce(self, label):
        """
        Announces the label if it's different from the last one spoken 
        to avoid repetitive talking.
        """
        if label != self.last_spoken:
            self.q.put(f"Best object detected: {label}")
            self.last_spoken = label

    def stop(self):
        self.stop_event.set()
        self.thread.join()
