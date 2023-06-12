import paho.mqtt.client as mqtt
import time

# MQTT broker details
broker = "agbc-fe.pdn.ac.lk"
port = 1883
username = "e18-team"
password = "pera@e18"

# Callback functions
def on_connect(client, userdata, flags, rc):
    print("Connected with result code " + str(rc))
    # Subscribe to topics after connection is successful
    client.subscribe("v0/controller/1000/blower")
    client.subscribe("v0/controller/1000/mist")
    print("kaarna")

def on_message(client, userdata, msg):
    print("Received message: " + msg.topic + " " + str(msg.payload))

    # Process the received message and take action accordingly
    if msg.topic == "v0/controller/1000/blower":
        if msg.payload == b'1':
            # Code for turning on the blower
            print("Blower turned on")
        elif msg.payload == b'0':
            # Code for turning off the blower
            print("Blower turned off")
        elif msg.topic == "v0/controller/1000/mist":
                if msg.payload == b'1':
                    # Code for turning on the mist
                    print("Mist turned on")
                elif msg.payload == b'0':
                    # Code for turning off the mist
                    print("Mist turned off")

# Create MQTT client and set callback functions
client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

# Set username and password for MQTT broker
client.username_pw_set(username, password)

# Connect to MQTT broker
client.connect(broker, port, 60)

# Start the MQTT client loop
client.loop_start()

# Testing loop
while True:
    # Publish messages with different payloads for testing
  
    client.publish("v0/controller/1000/blower", "0")
    time.sleep(2)
    print("akaa")
    client.publish("v0/controller/1000/mist", "0")
    time.sleep(2)