import socketio
import base64
import cv2
from PIL import Image
from io import BytesIO
import time
import os

class WebSocketClient:
    def __init__(self, channel):
        self.sio = socketio.Client()
        self.channel = channel
        self.frame = None
        self.url = os.environ["WS_URL"]
        self.initialize()

    def initialize(self):
        self.sio.connect(self.url)
        self.sio.emit("subscribe", {
            "channel": self.channel,
            "auth": {
                "headers": {
                    "Authorization": "Bearer " + os.environ["WS_API_KEY"],
                }
            }
        });

    def listen_for_events(self):
        print("Listening for events")
        @self.sio.on('new message')
        def new_message(channel, data):
            if self.channel == channel:
                print(f'Received event: {data}')

    def send_frames(self):
        while True:
            try:
                if self.frame is None:
                    continue

                img = Image.fromarray(cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB))
                buffered = BytesIO()
                img.save(buffered, 'jpeg')
                encoded_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

                if not self.sio.connected:
                    continue

                self.sio.emit("client event", {
                    "channel": self.channel,
                    "event": "new frame",
                    "data": {
                        "frame_b64": encoded_image,
                    },
                })

                time.sleep(0.033)
            except:
                self.initialize()

    def send_frame(self, frame):
        self.frame = frame
