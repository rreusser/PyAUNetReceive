import numpy as np


def hamming_window(n):
  return 0.54 - 0.46*np.cos(2.0*np.pi*np.arange(n)/(n-1.0))

def flattop_window(n):
  x = np.arange(n)/(n-1.0)
  return np.exp(-0.5*np.power((x-0.5)/0.4,8))


def absspectrum(y,window):
  y = np.abs(np.fft.rfft(y*window)) * np.arange(len(y)//2+1)
  #y[:10] *= 0
  return y

def summed_spec(l,r,window):
  sumspec = absspectrum(l,window) + absspectrum(r,window)
  return (sumspec + 1e-13) #* np.arange(len(sumspec))


def peak_search_in_range( freq, amp, fmin, fmax ):
  freq_filtered = np.where( (freq>fmin) & (freq<fmax) )
  imax = freq_filtered[0][ np.argmax(amp[freq_filtered]) ]
  fmax = freq[imax]
  amax = amp[imax]
  return (imax,fmax,amax)

def peak_search( freq, amp ):
  imax = np.argmax(amp[freq])
  fmax = freq[imax]
  amax = amp[imax]
  return (imax,fmax,amax)


class SignalMeta(object):
  '''A class to store and manage signal metadata. It performs lazy evaluation and caching of most
  things so that it's very efficient (beyond its use of getters/setters)'''
  def __init__(s, n=None, dt=None, nf=None, df=None ):
    '''Initialize from a vector and a time increment'''
    # Take whatever is provided from the parameters:
    s._n = n
    s._nf = nf
    s._dt = dt
    s._df = df

    # Initialize the rest:
    s._window_function = hamming_window
    s._window = None
    s._fbase = None
    s._tbase = None

  @property
  def fbase(s):
    '''Lazy evaluation of the frequency basis'''
    if s._fbase == None:
      s._fbase = np.arange(s.nf) * s.df
    return s._fbase

  @property
  def tbase(s):
    '''Lazy evaluation of the time basis'''
    if s._tbase == None:
      s._tbase = np.arange(s._n) * s._dt
    return s._tbase

  @property
  def window(s):
    '''Lazy evaluation of the window function basis'''
    if s._window == None:
      s._window = s._window_function(s._n)
    return s._window

  @property
  def n(s):
    if s._n == None:
      s._n = (s._nf-1)*2
    return s._n

  @n.setter
  def n(s,value):
    s._n = value
    s._fbase = None
    s._tbase = None
    s._window = None
    s._nf = None
    s._df = None


  @property
  def nf(s):
    if s._nf == None:
      s._nf = s._n//2 + 1
    return s._nf

  @nf.setter
  def nf(s,value):
    s._nf = value
    s._n = None
    s._fbase = None
    s._tbase = None
    s._window = None



  @property
  def dt(s):
    if s._dt == None:
      s._dt = 1.0 / (s.n * s.df)
    return s._dt

  @dt.setter
  def dt(s,value):
    s._dt = value
    s._fbase = None
    s._tbase = None
    s._df = None

  @property
  def df(s):
    if s._df == None:
      s._df = 1.0 / (s.n * s.dt)
    return s._df

  @df.setter
  def df(s,value):
    s._df = value
    s._fbase = None
    s._tbase = None
    s._dt = None




class Signal(SignalMeta):
  def __init__(s, n=None, dt=None, nf=None, df=None ):
    SignalMeta.__init__(s,n=n,dt=dt,nf=nf,df=df)
    s._y = None
    s._yf = None

    # Plot references
    s.tp = None
    s.fp = None

  @classmethod
  def from_time_domain(cls, y=None, dt=None, n=None, nf=None, df=None):
    s = Signal(dt=dt, n=n, nf=nf, df=df)
    if y == None:
      s._n = n
      s._y = np.zeros(n,dtype='d')
    else:
      s.y = y
    return s

  @classmethod
  def from_freq_domain(cls, yf=None, df=None, nf=None, n=None, dt=None):
    s = Signal(df=df, nf=nf, n=n, dt=dt)
    if yf == None:
      s._nf = nf
      s._yf = np.zeros(nf,dtype='complex128')
    else:
      s.yf = yf
    return s

  @property
  def y_data(s):
    return s._y

  @y_data.setter
  def y_data(s,values):
    if s._y == None:
      s._y = values
    else:
      s._y[:] = values
    s._yf = None

  def set_y_data(s, y):
    if s._y == None:
      s._y = y
    else:
      s._y[:] = y
    s._yf = None

  def set_yf_data(s,yf):
    if s._yf == None:
      s._yf = yf
    else:
      s._yf[:] = yf
    s._y = None

  @property
  def yf_data(s):
    return s._yf

  @yf_data.setter
  def yf_data(s,values):
    if s._yf == None:
      s._yf = values
    else:
      s._yf[:] = values
    s._y = None

  @property
  def y(s):
    if s._y == None:
      s._y = np.fft.irfft(s._yf)
    return s._y

  @y.setter
  def y(s,value):
    s._y = value
    s.n = len(s._y)
    s._yf = None

  @property
  def yf(s):
    if s._yf == None:
      s._yf = np.fft.rfft(s._y)
    return s._yf

  @yf.setter
  def yf(s,value):
    s._yf = value
    s.nf = len(s._yf)
    s._y = None

  def touch_y(s):
    s._yf = None

  def touch_yf(s):
    s._y = None

  def plot(s, live=False):
    s.tplot(live)

  def tplot(s, live=False, show=False):
    if live:
      pl.ion()
    if s.tp==None or not live:
      s.tp = pl.plot(s.tbase, s.y)[0]
    else:
      s.tp.set_ydata( s.y )
    pl.axis([np.min(s.tbase),np.max(s.tbase),np.min(s.y),np.max(s.y)])
    if live:
      pl.draw()
    elif show:
      pl.show()

  def fplot(s, live=False, show=False):
    if live:
      pl.ion()
    plotdata = np.abs(s.yf)
    if s.fp==None or not live:
      s.fp = pl.plot(s.fbase, plotdata)[0]
    else:
      s.fp.set_ydata(np.abs(s.yf))
    pl.axis([np.min(s.fbase),np.max(s.fbase),np.min(plotdata),np.max(plotdata)])
    if live:
      pl.draw()
    elif show:
      pl.show()

  def tfplot(s, live=False, tmax=None, fmax=None, show=False):
    if live:
      pl.ion()
    pl.subplot(2,1,1)

    if s.tp==None:
      s.tp = pl.plot(s.tbase, s.y)[0]
    else:
      s.tp.set_ydata( s.y )
    pl.axis([np.min(s.tbase),tmax or np.max(s.tbase),np.min(s.y),np.max(s.y)])


    pl.xlabel('Time, t, s')
    pl.subplot(2,1,2)

    plotdata = np.abs(s.yf)
    if s.fp==None or not live:
      s.fp = pl.plot(s.fbase, plotdata)[0]
    else:
      s.fp.set_ydata(np.abs(s.yf))
    pl.axis([np.min(s.fbase),fmax or np.max(s.fbase),np.min(plotdata),np.max(plotdata)])


    pl.xlabel('Frequency, f, Hz')

    if live:
      pl.draw()
    elif show:
      pl.show()




class AudioBuffer:
  '''Eats audio data and executes a callback when a specified amount of data is
  accumulated.'''

  def __init__(s, num_bytes=1024, callback=lambda x: None):
    s.buf = ''
    s.bufsize = num_bytes
    s.callback = callback

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

  where t_0 is the half-life of the memory.'''
  def __init__(s):
    '''Initialize the beat detector. Screwing with the config is up to you.'''

    # The number of running samples with which to compare the new sample
    s.n = 256


    # Parameters for the incoming data:

    # Audio sample rate:
    s.sample_freq = 44100
    
    # The number of samples considered at a time in the audio input:
    s.window_width = 1024

    # Just precompute this since we reference it a bunch:
    s.window_halfwidth = s.window_width//2

    # Precalculate a window function for the fft:
    s.window = hamming_window(s.window_width)

    # Width of a window of data, in seconds:
    s.window_dt = 1.0*s.window_width / s.sample_freq

    s.spectrum = Signal.from_time_domain(n=s.n, dt=s.window_dt)
    s.cepstrum = Signal.from_time_domain(n=s.n*8, dt=s.window_dt)

    s.beat_spectrum_window = flattop_window(s.spectrum.n)

    # A reference to the function with which we compute the spectrum of each segment
    # of data:
    s.spectrum_function = summed_spec

    # There's really no sense storing a full similarity matrix. Instead, let's just
    # accumulate data into a running beat spectrum equivalent to diagonal sums of the
    # similarity matrix. Of course since we're not storing the whole thing, and since
    # it's a realtime algorithm, we'll have to do a decaying sort of sum.
    s.diagonal_sums = np.zeros(s.n)

    s.circular_buffer = np.zeros((s.n,s.window_width//2+1))
    s.circular_buffer_position = 0

    # Characteristic decay time of the beat spectrum:
    s.halflife = 0.5

    # The window shifts by half of the window width due to the half-window overlap,
    # so the time constant accounts for this:
    s.decay_factor = np.exp( -np.log(2.0) * ( s.window_dt * 0.5 ) / s.halflife )

    s.l_halfprev = None
    s.r_halfprev = None

    # Precalculate the matrices necessary for a linear fit:
    s.A = np.vander(np.arange(s.n),3)
    s.At = np.linalg.pinv(s.A)

    # Time-average the cepstrum to suppress fluctuations
    s.time_averaged_cepstrum = Signal.from_time_domain(n=s.cepstrum.n, dt=s.window_dt)
    s.time_averaged_cepstrum.yf

  def accumulate(s,l,r):
    '''Assimilate new stereo input into the running beat spectrum. This function expects
    samples that are an integral multiple of the window width, and frankly it'd probably
    be a really smart thing to do to just use a power of two.'''
    if s.l_halfprev==None or s.r_halfprev==None:
      offset = 0
    else:
      offset = -s.window_halfwidth

    samples = len(l)

    # Let's not hard-code the window width so that you'd be able to split up the samples
    # into smaller chunks in case the audio input has too many samples at a time and
    # splitting it by just a factor of two results in segments that are too long.
    #
    # So shift the offset, starting with a window that has half old samples and half
    # new samples:
    while offset + s.window_width <= samples:

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
    correlation = np.roll( rolled_correlation[::-1], s.circular_buffer_position+1 )

    # The correlations are kind of all over the place, so subtract off a linear fit
    coeff = np.dot(s.At,correlation)
    fit = np.dot(s.A,coeff)
    correlation -= fit

    # Decay the existing spectrum and add the new:
    s.spectrum._y *= s.decay_factor
    s.spectrum._y += correlation
    s.spectrum.touch_y()

    s.circular_buffer_position = (s.circular_buffer_position+1)%s.n


  def update_cepstrum(s):
    # Return the frequency-domain autocorrelation of the beat spectrum:
    yh = np.fft.rfft(s.spectrum.y * s.beat_spectrum_window,s.cepstrum.n) * np.arange(s.cepstrum.n//2+1,dtype='d') / s.cepstrum.n
    s.cepstrum.set_yf_data( np.abs( yh * np.conj(yh) ) )

    s.time_averaged_cepstrum._yf *= s.decay_factor
    s.time_averaged_cepstrum._yf += s.cepstrum._yf
    s.time_averaged_cepstrum.touch_yf()




  def maxima(s,cutoff=None):

    # Apply the cutoff frequency:
    icutoff = int(cutoff / s.time_averaged_cepstrum.df) if cutoff else len(s.time_averaged_cepstrum.yf)

    # Locate maxima:
    imax = (np.diff(np.sign(np.diff(s.time_averaged_cepstrum._yf[:icutoff]))) < 0).nonzero()[0] + 1

    # Evaluate frequency and amplitudes at the maxima:
    fmax = s.time_averaged_cepstrum.fbase[imax]
    amax = s.time_averaged_cepstrum.yf[imax]
    
    # Sort by amplitude descending:
    o = np.argsort(amax**2)[::-1]

    return (imax[o], fmax[o], amax[o])
    


class TempoDetector:
  def __init__(s):

    # The tempo detector owns a beat spectrum with which it maintains a beat spectrum:
    s.beats = BeatDetector()

    # Arrays to store the points of interest:
    s.freq_peaks = []
    s.amp_peaks = []

    s.halflife = 0.05
    s.decay_factor = np.exp( -np.log(2.0) * ( s.beats.window_dt * 0.5 ) / s.halflife )

    s.bpm = 0
    

  def accumulate(s,l,r):
    s.beats.accumulate(l,r)
    s.beats.update_cepstrum()

    s.track_bpm()


  def track_bpm(s):
    m = s.beats.maxima()
    bpm = m[1][0]*60

    s.bpm *= s.decay_factor
    s.bpm += (1.0-s.decay_factor)*bpm

    print round(s.bpm,1)

    



if __name__ == "__main__":

  from pyaunetreceive import AUNetReceive
  import matplotlib.pyplot as pl

  tempo_detector = TempoDetector()
  beats = tempo_detector.beats


  pl.ion()
  g = pl.plot([],'.')[0]

  frame = 0
  def process_buffered_data(data):
    global tempo_detector, beats, frame, g
    lr = np.fromstring(data,dtype=np.int16)/32768.0
    l = lr[::2]
    r = lr[1::2]
    tempo_detector.accumulate(l,r)

    # Plot:
    if frame%4==0:

      beats.time_averaged_cepstrum.tfplot(live=True, tmax=10)
      #beats.cepstrum.tfplot(live=True, tmax=10)
      #beats.spectrum.tplot(live=True)
      pass

    frame+=1


  audiobuffer = AudioBuffer(4096, process_buffered_data)


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

    beats.cepstrum.tfplot(tmax=10, fmax=20)
    (imax, fmax, amax) = beats.maxima(10)
    pl.plot( beats.cepstrum.fbase[imax[:5]], beats.cepstrum.yf[imax[:5]], '.')
    pl.show()
