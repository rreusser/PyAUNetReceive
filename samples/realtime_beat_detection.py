import numpy as np


def hamming_window(n):
  return 0.54 - 0.46*np.cos(2.0*np.pi*np.arange(n)/(n-1.0))

def absspectrum(y,window):
  return np.abs(np.fft.rfft(y*window))

def summed_spec(l,r,window):
  sumspec = absspectrum(l,window) + absspectrum(r,window)
  return np.log(sumspec +1e-13) #* np.arange(len(sumspec))


def peak_search( freq, amp, fmin, fmax ):
  freq_filtered = np.where( (freq>fmin) & (freq<fmax) )
  imax = freq_filtered[0][ np.argmax(amp[freq_filtered]) ]
  fmax = freq[imax]
  amax = amp[imax]
  return (imax,fmax,amax)



class AudioBuffer:
  '''Eats audio data and executes a callback when a specified amount of data is
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




class BeatSpectrum:
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

    # A reference to the function with which we compute the spectrum of each segment
    # of data:
    s.spectrum_function = summed_spec
    
    # The number of running samples with which to compare the new sample
    s.n = 512

    # This is the beat spectrum in which we accumulate the data!
    s.beat_spectrum = np.zeros(s.n)
    s.beat_spectrum_window = hamming_window(s.n)

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
    s.diagonal_sums = np.zeros(s.n)

    s.circular_buffer = np.zeros((s.n,s.window_width//2+1))
    s.circular_buffer_position = 0

    # Audio sample rate:
    s.sample_freq = 44100

    # Characteristic decay time of the beat spectrum:
    s.halflife = 2.5

    # Width of a window of data, in seconds:
    s.window_dt = 1.0*s.window_width / s.sample_freq

    s.dt = 0.5*s.window_dt
    s.df = 1.0 / (s.n * s.dt)

    # The window shifts by half of the window width due to the half-window overlap,
    # so the time constant accounts for this:
    s.decay_factor = np.exp( -np.log(2.0) * ( s.window_dt * 0.5 ) / s.halflife )

    s.l_halfprev = None
    s.r_halfprev = None

    # Precalculate the matrices necessary for a linear fit:
    s.A = np.vander(np.arange(s.n),3)
    s.At = np.linalg.pinv(s.A)


  def accumulate(s,l,r):
    '''Assimilate new stereo input into the running beat spectrum. This function expects
    samples that are an integral multiple of the window width, and frankly it'd probably
    be a really smart thing to do to just use a power of two.'''
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

    # The correlations are kind of all over the place, so subtract off a linear fit
    coeff = np.dot(s.At,correlation)
    fit = np.dot(s.A,coeff)
    correlation -= fit

    # Decay the existing spectrum and add the new:
    s.beat_spectrum *= s.decay_factor
    s.beat_spectrum += correlation

    s.circular_buffer_position = (s.circular_buffer_position+1)%s.n


  def beat_cepstrum(s, n):
    # Window the beat spectrum:
    y = s.beat_spectrum * s.beat_spectrum_window

    # Return the zero-padded fft:
    return np.fft.rfft( y, n )


class TempoDetector:
  def __init__(s):

    # The tempo detector owns a beat spectrum with which it maintains a beat spectrum:
    s.beat_spectrum = BeatSpectrum()

    # Store a quick reference because I'm sick of typing beat_spectrum
    s.bs = s.beat_spectrum

    # The number of time-domain samples:
    s.nt = s.bs.n

    # A window for the beat spectrum:
    s.window = hamming_window(s.nt)

    # The factor by which to upsample the beat spectrum when taking the fft:
    s.upsample = 4

    # The number of frequency-domain samples:
    s.nf = (s.bs.n*s.upsample)//2+1

    # Calculate the frequency increment for the fft'd beat spectrum
    s.df = s.bs.df / s.upsample

    # Calculate the frequency basis:
    s.fbase = np.arange((s.bs.n*s.upsample)//2+1) * s.df
    s.bpm = s.fbase * 60


    # Arrays to store the points of interest:
    s.freq_peaks = []
    s.amp_peaks = []
    

  def accumulate(s,l,r):
    s.bs.accumulate(l,r)

  def beat_cepstrum(s):
    return np.abs(s.bs.beat_cepstrum( s.bs.n * s.upsample ))

  def peaks(s):
    cep = s.beat_cepstrum()

    # The cepstrum appears to have strong harmonics, so take the autocorrelation as a way
    # to get at the periodicity of these harmonics... maybe.



    if False:
      bpm_max = np.max(s.bpm)

      # This is ad hoc and not strictly 'correct' for the peak half-magnitude width, but
      # the important thing is that it's useful and scales correctly.
      peak_width = np.pi / s.bs.n / s.bs.window_dt * 60.0

      # Find the peak in a nice range:
      imax, fmax, amax = peak_search( s.bpm, cep, 150, 450 )

      # Remove this peak from the spectrum:
      cep *= 1.0 - np.exp( -(s.bpm - fmax)**2*0.5/peak_width**2 )

      pl.plot(s.bpm, cep)
      pl.show()


    
    



if __name__ == "__main__":

  from pyaunetreceive import AUNetReceive
  import matplotlib.pyplot as pl


  tempo_detector = TempoDetector()
  beat_spectrum = tempo_detector.beat_spectrum

  pl.ion()

  plot_cepstrum = True
  live_plots = True
  g = None


  frame = 0
  def process_buffered_data(data):
    global g, frame
    lr = np.fromstring(data,dtype=np.int16)/32768.0
    l = lr[::2]
    r = lr[1::2]
    tempo_detector.accumulate(l,r)
    if frame%2 == 0 and live_plots:
      if plot_cepstrum:
        yh = (np.abs(tempo_detector.beat_cepstrum()))
        xh = tempo_detector.fbase * 60
        pl.xlabel('Beats per minute, bpm')
        pl.axis([0,1000,np.min(yh[1:]),np.max(yh[1:])])
      else:
        yh = tempo_detector.beat_spectrum.beat_spectrum[::-1]
        xh = np.arange(len(yh)) * tempo_detector.beat_spectrum.dt
        pl.xlabel('Time, t, seconds')
        pl.axis([0,np.max(xh),np.min(yh[1:]),np.max(yh[1:])])
      g = g or pl.plot(xh, np.arange(len(yh)))[0]
      g.set_ydata( yh )
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
    pass
    if False:
      td = tempo_detector
      bsa = tempo_detector.beat_spectrum
      cep = tempo_detector.beat_cepstrum()
      pl.plot(tempo_detector.fbase * 60, cep )
      pl.axis([0,1000,0,np.max(cep)])
      pl.show()
