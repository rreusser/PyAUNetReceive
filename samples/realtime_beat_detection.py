import numpy as np


def hamming_window(n):
  return 0.54 - 0.46*np.cos(2.0*np.pi*np.arange(n)/(n-1.0))

def absspectrum(y,window):
  global debug
  d = np.abs(np.fft.rfft(y*window))
  #d -= np.mean(d)
  return d

def summed_spec(l,r,window):
  global debug
  sumspec = absspectrum(l,window) + absspectrum(r,window)
  return np.log(sumspec +1e-13) #* np.arange(len(sumspec))


debug = None


class AudioBuffer:
  '''Eats audio data and executes a callback when a certain amount of data is
  accumulated.'''

  def __init__(s, num_bytes=1024):
    s.buf = ''
    s.bufsize = num_bytes
    s.callback = lambda x: None

  def buffer(s,data):
    s.buf += data
    if len(s.buf) >= s.bufsize:
      s.callback( s.buf[0:s.bufsize] )
      s.buf = s.buf[s.bufsize:]


class BeatDetector:
  '''The basis for this algorithm is the paper "The Beat Spectrum: A New Approach to
  Rhythm Analysis" by Foote and Uchihashi. Their algorithm is:

    1) Compute the log-magnitude spectrum of 256-sample-windowed segments of data
       (with half-window overlap).

    2) Take the spectra of, say, 256 adjacent segments and compute the correlation
       using a function for spectra v1 and v2:

                        v1 * v2
                    ---------------
                    ||v2|| * ||v2||

    3) Add up the diagonals of the similarity matrix. This is the beat spectrum. Peaks
       correspond to the repetitive patterns in the spectrum.

  My algorith is similar, except I've adapted it for realtime analysis. Instead of
  computing a static similarity matrix, I'm opting for tracking only the running
  diagonal sums of the similarity matrix. Of course an infinite sum diverges and never
  forgets, so I'm letting old terms decay exponentially.  Every time a new correlation
  is added is added, the existing summation is multiplied by a factor close to 1.
  To be precise,
  
                  sum = sum * exp(-alpha*dt)

  with

                  alpha = log(0.5) / t_0

  where t_0 is the half-life of the memory.
  
  OMG if this ever works it's gonna be a freakin miracle.'''
  def __init__(s):
    '''Initialize the beat detector. Screwing with the config is up to you.'''

    # A reference to the function with which we compute the spectrum of each segment
    # of data:
    s.spectrum_function = summed_spec
    
    # The number of running samples with which to compare the new sample
    s.memory = 512

    # This is the beat spectrum in which we accumulate the data!
    s.beat_spectrum = np.zeros(s.memory)

    # The number of samples considered at a time in the audio input:
    s.window_width = 1024

    # The number of samples expected in the audio input:
    s.samples_per_input = 1024

    # Precalculate a window function for the fft:
    s.window = hamming_window(s.window_width)

    # Just precompute this since we reference it a bunch:
    s.window_halfwidth = s.window_width//2

    # There's really no sense storing a full similarity matrix. Instead, let's just
    # accumulate data into a running beat spectrum equivalent to diagonal sums of the
    # similarity matrix. Of course since we're not storing the whole thing, and since
    # it's a realtime algorithm, we'll have to do a decaying sort of sum.
    s.diagonal_sums = np.zeros(s.memory)

    s.circular_buffer = np.zeros((s.memory,s.window_width//2+1))
    s.circular_buffer_position = 0

    # Audio sample rate:
    s.sample_freq = 44100

    # Characteristic decay time of the beat spectrum:
    s.halflife = 2.5

    # Width of a window of data, in seconds:
    s.window_dt = 1.0*s.window_width / s.sample_freq

    # The window shifts by half of the window width due to the half-window overlap,
    # so the time constant accounts for this:
    s.decay_factor = np.exp( -np.log(2.0) * ( s.window_dt * 0.5 ) / s.halflife )

    s.l_halfprev = None
    s.r_halfprev = None

  def assimilate(s,l,r):
    global debug
    '''Assimilate new stereo input into the running beat spectrum. This function expects
    samples that are an integral multiple of the window width, and frankly it'd probably
    be a really smart thing to do to just use a power of two. Srsly.'''
    if s.l_halfprev==None or s.r_halfprev==None:
      offset = 0
    else:
      offset = -s.window_halfwidth

    # Let's not hard-code the window width so that you'd be able to split up the samples
    # into smaller chunks in case the audio input has too many samples at a time and
    # splitting it by just a factor of two results in segments that are too long.
    #
    # So shift the offset, starting with a window that has half old samples and half
    # new samples:
    while offset + s.window_width <= s.samples_per_input:

      if offset < 0:
        # In this case, it's split between old samples and new samples, so we'll grab
        # a bit of both and concatenate:
        l_input = np.concatenate( (s.l_halfprev, l[:offset+s.window_width]) )
        r_input = np.concatenate( (s.r_halfprev, r[:offset+s.window_width]) )

      else:
        # In this case, it's all new data:
        l_input = l[offset:offset+s.window_width]
        r_input = r[offset:offset+s.window_width]

      # Once we have our input data, apply the spectrum function to it:
      spec = s.spectrum_function(l_input,r_input, s.window)

      # Our similarity metric between spectra A and B is:
      #
      #   ( A*B ) / ( ||A|| * ||B|| ),
      #
      # so we'll just start off by by dividing each vector by it's own norm, then when
      # we compute the similarity, we'll only have to calculate and divide by the norm
      # of the new vector.
      spec /= np.linalg.norm(spec)

      # Okay, so 'spec' is the new spectrum that's being assimilated into the beat
      # detector's current state:
      s._append_to_spectra( spec )

      # Finally, shift the window and do the whole thing again if there are more sample
      # windows available in this audio data:
      offset += s.window_halfwidth


    # Store a reference to the previous data so that we can half-overlap the previous
    # window's data
    s.l_halfprev = l[-s.window_halfwidth:]
    s.r_halfprev = r[-s.window_halfwidth:]


  def _append_to_spectra(s,v):
    '''Once we've calculated the spectrum of the window, assimilate it into the running
    diagonal sums. This method uses a circular buffer that so the we can keep our shit
    very nicely vectorized. Wow this is a horrible case of premature optimization.'''

    # Okay this is where it gets freakin complicated. There's a chance this approach
    # would benefit from a transpose, but I just don't remember whether these are row-major
    # or column-major arrays. What am I doing with my life?
    s.circular_buffer[s.circular_buffer_position,:] = v

    # Now correlate the current spectrum with each element of the circular buffer.
    # Note: uses broadcasting to perform per-column multiplication of a vector without
    # actually tiling the comparison vector into a full matrix. Also note that it's all
    # real magnitude data so we can neglect the complex conjuwhatnot.
    rolled_correlation = np.dot( s.circular_buffer, v ) / np.linalg.norm(v)

    # The correlation has an offset determined by the cirular buffer's current position,
    # so we'll roll the array so that the first element of the correlations is always
    # the autocorrelation:
    correlation = np.roll( rolled_correlation, -s.circular_buffer_position-1 )

    s.beat_spectrum *= s.decay_factor
    s.beat_spectrum += correlation
    
    s.circular_buffer_position = (s.circular_buffer_position+1)%s.memory





if __name__ == "__main__":

  from pyaunetreceive import AUNetReceive
  import matplotlib.pyplot as pl

  debug = np.zeros(0)


  beat_detector = BeatDetector()

  pl.ion()
  g = pl.plot(np.arange(beat_detector.memory))[0]


  frame = 0
  def process_buffered_data(data):
    global g, frame
    lr = np.fromstring(data,dtype=np.int16)/32768.0
    l = lr[::2]
    r = lr[1::2]
    beat_detector.assimilate(l,r)
    if frame%2 == 0:
      y = beat_detector.beat_spectrum[::-1]
      g.set_ydata(y)
      pl.axis([0,beat_detector.memory,np.min(y),np.max(y)])
      pl.draw()
    frame += 1


  audiobuffer = AudioBuffer(4096)
  audiobuffer.callback = process_buffered_data


  def process_audio_data(data):
    global audiobuffer
    audiobuffer.buffer(data)





  try:
    if True:
      netrecv = AUNetReceive( host='127.0.0.1', port=52800 )
      netrecv.onrecv( process_audio_data )
      netrecv.listen()

    else:
      f = open('audiodata.pcm','r')
      d = f.read()
      for i in range(0,1600):
        process_audio_data( d[1024*i:1024*(i+1)] )
      raise KeyboardInterrupt

  except KeyboardInterrupt:
    if False:
      pl.show()
      b=beat_detector
      pl.plot(b.beat_spectrum)
