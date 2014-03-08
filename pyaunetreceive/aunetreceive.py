import socket, select, re

class AUNetReceive:

  STATE_NEEDSHAKE=0
  STATE_NEEDMETA=1
  STATE_WAITFORBEGIN=2
  STATE_EXPECTSYNC=3
  STATE_EXPECTDATA=4

  def __init__(s, host='localhost', port=52800):
    '''Initialize AUNetReceive and connect to the specified address'''
    s.state = s.STATE_NEEDSHAKE
    s.data_queue = ''
    s.netrecv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.netrecv.connect((host,port))
    s.netrecv.setblocking(1)
    s.callback = lambda x: None
    
  def listen(s):
    '''Initiate listening for audio on the socket'''
    while 1:
      try:
        while s._process_data_queue():
          pass
      except RuntimeError:
        pass

      for sock in select.select([s.netrecv],[],[])[0]:
        if sock==s.netrecv:
          s.data_queue += sock.recv(4096)

  def onrecv(s,callback):
    '''Provide a callback which will process the raw interleaved audio data'''
    s.callback = callback

  def _process_data_queue(s):
    '''Empty out the current queue of data, to the extent possible'''
    if len(s.data_queue) == 0:
      return False

    try:
      if s.state == s.STATE_NEEDSHAKE:
        d = s._pop(16)
        s._shake(d)
      elif s.state == s.STATE_NEEDMETA:
        d = s._pop(40)
        s._parse_meta(d)
      elif s.state == s.STATE_WAITFORBEGIN:
        d = s._pop(4)
        s._verify_begin(d)
      elif s.state == s.STATE_EXPECTSYNC:
        d = s._pop(4)
        s._sync(d)
      elif s.state == s.STATE_EXPECTDATA:
        d = s._pop(1024)
        s._data(d)

      return True if len(s.data_queue)>0 else False

    except RuntimeError:
      # If error, then let's put the string back and wait for more data
      s.data_queue = d + s.data_queue
      s._recover()


  def _shake(s,data):
    '''Perform handshake with the AUNetSend plugin instance'''
    if re.compile('^ausend').search(data) is not None:
      s.netrecv.send( 'aurecv'.ljust(40,'\0') )
      s.state = s.STATE_NEEDMETA
    else:
      raise RuntimeError

  def _parse_meta(s,data):
    '''Parse metadata (simply drops the data since it's all PCM int16 data right now)'''
    s.state = s.STATE_WAITFORBEGIN

  def _verify_begin(s,data):
    '''Passed by AUNetSend before initiating stream'''
    if data == '\0\0\0\0':
      s.state = s.STATE_EXPECTSYNC
    else:
      raise RuntimeError

  def _sync(s,data):
    '''Sent between 1024-byte data blocks'''
    if data == 'sync':
      s.state = s.STATE_EXPECTDATA
    else:
      raise RuntimeError

  def _data(s,data):
    '''Execute the callback on the data'''
    s.callback(data)
    s.state = s.STATE_EXPECTSYNC

  # Discard data until we find a 'sync'
  def _recover(s):
    '''Attempt to discard data until we find a sync block'''
    s._pop( s.data_queue.find('sync') )
    
  def _pop(s,numchars):
    '''Pop and return a specified number of characters from the data queue'''
    popped = s.data_queue[0:numchars]
    s.data_queue = s.data_queue[numchars:]
    return popped
    

if __name__ == "__main__":
  netrecv = AUNetReceive()
  netrecv.listen()
