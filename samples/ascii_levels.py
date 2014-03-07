import numpy as np
from pyaunetreceive import AUNetReceive


def process_audio_data(data):
  lr = np.fromstring(data,dtype=np.int16)
  l = lr[::2]
  r = lr[1::2]
  var = np.var(l) + np.var(r)
  print '*'*int(var/1000000)


netrecv = AUNetReceive( host='127.0.0.1', port=52801 )
netrecv.onrecv( process_audio_data )
netrecv.listen()
