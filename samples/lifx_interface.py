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
    s.brightness = 0.2
    s.last_updated_at = 0
    s.last_hue_jump_at = 0
    s.max_brightness = 0.2

    s.recent_changes = np.zeros(0)
    s.recent_hue_jumps = np.zeros(0)

    try:
      s.lifx_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
      s.lifx_socket.connect(('localhost', 8080))
    except socket.error:
      s.lifx_socket = None
      print 'No lifx-server detected. :('

  def hue_jump(s):
    t = time.time()
    s.last_hue_jump_at = t
    s.recent_hue_jumps = np.append(s.recent_hue_jumps,[t])
    s.hue = (s.hue + 60+240*np.random.rand())

  def update(s, fadeTime=300 ):
    t = time.time()
    s.last_updated_at = t
    if s.lifx_socket==None:
      return
    s.lifx_socket.send(json.dumps(s._update_json(fadeTime))+'\n')

  def time_since_last_update(s):
    return time.time() - s.last_updated_at

  def time_since_last_jump(s):
    return time.time() - s.last_hue_jump_at

  def _update_json(s, fadeTime=300):
    return {
      'operation': 'color',
      'value': {
        'hue': s.hue,
        'brightness': s.brightness*s.max_brightness,
        'saturation': s.sat,
        'fadeTime': fadeTime
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
accum = []
spec = []
recent_samples = [0,0,0,0,0]
last_light_change_at = 0


lifx = LIFX()

def window(n):
  return 0.54 - 0.46*np.cos(math.pi*2*np.arange(n)/(n-1))

frame = 0

def process_audio_data(data):
  
  global slow_meter, fast_meter, accum, num_accum, cur_accum, spec, max_spec, recent_samples, last_light_change_at, lifx, frame

  frame += 1

  # Convert to array
  lr = np.fromstring(data,dtype=np.int16)/32768.0

  accum.append(lr)

  while len(accum) > 5:
    accum = accum[1:]

  lr = np.concatenate(accum)

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
  while len(recent_samples) > 20:
    recent_samples = recent_samples[1:]


  peaked = (recent_samples[-2] > np.max(recent_samples[-6:-2]) and recent_samples[-1] < recent_samples[-2])


  if lifx.time_since_last_jump() > 0.25 and peaked and bigness > 1.5:
    #print 'peaked'
    lifx.brightness = 1.0
    lifx.hue_jump()
    lifx.update(0)
  else:
    #hue_inc = fast_meter.level
    #lifx.hue += np.min([hue_inc,1.0])

    if lifx.time_since_last_update() > 0.5:
      #print '******************************'
      old_brightness = lifx.brightness*1.0
      lifx.brightness *= 0.8
      lifx.update( (50 if lifx.brightness/old_brightness > 1.2 else 300) )
      #old_brightness = lifx.brightness*1.0
      #lifx.brightness = 0.25 + 0.75*np.clip(max_recent_bigness/3,0,1)
      #lifx.update( (50 if lifx.brightness/old_brightness > 1.2 else 300) )


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
