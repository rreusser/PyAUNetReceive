import json
import socket
import math
import time
import numpy as np
from pyaunetreceive import AUNetReceive
import matplotlib.pyplot as pl


# Connect to a running instance of lifx-server

def send_to_lifx(command):
  lifx.send(json.dumps(command)+'\n')


class LIFX:
  def __init__(s):
    s.hue = 0.0
    s.sat = 1.0
    s.bright = 0.05

    try:
      s.lifx_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
      s.lifx_socket.connect(('localhost', 8080))
    except socket.error:
      s.lifx_socket = None
      print 'No lifx-server detected. :('

  def randomize_hue(s):
    s.hue = np.random.rand()*360.0

  def update(s):
    if s.lifx_socket==None:
      return
    s.lifx_socket.send(json.dumps(s._update_json())+'\n')

  def _update_json(s):
    return {
      'operation': 'color',
      'value': {
        'hue': s.hue,
        'brightness': s.bright,
        'saturation': s.sat,
        'fadeTime': 200
      }
    }



class LevelMeter:
  def __init__(s, halflife=1.0, sample_rate=44100):
    s.constant = -math.log(2) / halflife
    s.sample_dt = 1.0/sample_rate
    s.level = 1.0
  
  # Include a level coming from a number of samples:
  def accum(s, cur_level, num_samples):
    dt = num_samples * s.sample_dt
    decay = math.exp(dt*s.constant)
    s.level *= decay
    s.level += (1.0 - decay) * cur_level



slow_meter = LevelMeter( 5.0, 44100 )
fast_meter = LevelMeter( 0.01, 44100 )
var_meter = LevelMeter( 5.0, 44100 )

num_accum = 2
cur_accum = 0
max_spec = 1000
accum = np.zeros(0)
spec = []
recent_samples = [0,0,0,0,0]
last_light_change_at = 0


lifx = LIFX()

def window(n):
  return 0.54 - 0.46*np.cos(math.pi*2*np.arange(n)/(n-1))

def process_audio_data(data):
  
  global slow_meter, fast_meter, accum, num_accum, cur_accum, spec, max_spec, recent_samples, last_light_change_at, lifx

  # Convert to array
  lr = np.fromstring(data,dtype=np.int16)/32768.0

  if cur_accum < num_accum:
    accum = np.append(accum,lr)
    cur_accum += 1
    return
  else:
    lr = accum*1
    accum = np.zeros(0)
    cur_accum = 0

  # Separate the channels
  l = lr[::2]
  r = lr[1::2]

  # Count samples
  n = len(l)

  w = window(len(l))
  lh = np.fft.rfft(l*w)
  rh = np.fft.rfft(r*w)


  kernel = np.linspace(0,1,len(lh))**0.5
  fftabs = np.abs(lh * kernel) + np.abs(rh * kernel)

  if len(spec) < max_spec:
    spec.append(fftabs)

  std = np.linalg.norm(fftabs[len(fftabs)//10:])
  var = np.square(std)

  # Accumulate this data into the level meters
  slow_meter.accum( std, n )
  fast_meter.accum( std, n )

  # Calculate the local contrast in sound
  local_diff = fast_meter.level - slow_meter.level

  # Track local contrast level
  var_meter.accum( local_diff**2, n )

  # Calculate the magic parameter
  bigness = local_diff / np.sqrt(var_meter.level)

  recent_samples.append(bigness)
  while len(recent_samples) > 5:
    recent_samples = recent_samples[1:]



  time_since_last_update = time.time() - last_light_change_at

  if time_since_last_update > 0.5 and recent_samples[-1] > recent_samples[-2] and recent_samples[-1] > 3.5:
    lifx.randomize_hue()
    lifx.update()
    last_light_change_at = time.time()

  if time_since_last_update > 0.3:
    pass

  

  #levelmeter = std
  #levelmeter = slow_meter.level
  #levelmeter = fast_meter.level
  #levelmeter = 20 + local_diff
  levelmeter = 20 + 20*bigness


  print u'\u2588'*int( levelmeter )




try:
  netrecv = AUNetReceive( host='127.0.0.1', port=52800 )
  netrecv.onrecv( process_audio_data )
  netrecv.listen()
except KeyboardInterrupt:

  if False:
    pl.imshow(np.flipud(np.transpose(np.array(spec))))
    pl.show()
