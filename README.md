# PyAUNetReceive

PyAUNetReceive is a simple Python client that listens on a socket for output from the AUNetSend Audio Unit plugin. Translation: If you add the AUNetSend plugin to GarageBand or something, you can process the audio in realtime and do with it anything Python can do! Sweet!

## Installation

    $ python setup.py install

## Usage

To connect to the output of an AUNetSend plugin, first start a program the uses Audio Unit plugins and then instantiate an AUNetSend plugin. Signed 16-bit PCM output is the only format currently supported. Specify a host and port (default host='127.0.0.1', port=52800) in the Audio Unit view.

Apple's AU Lab is the simplest way to get started. You can find AU Lab with a quick search on [Apple's Developer Site](https://developer.apple.com/downloads/index.action). For a quick start,

1. Select Add Audio Unit Generator from the Edit menu. Select AUAudioFilePlayer, click OK, and use the dialog to find and play an audio file.

2. Find the Effects section of the Output pane and select AUNetSend. Set the data format to '16 bit integer PCM' and the port to 52800.

Other suggested programs are the excellent [Audio Hijack](http://www.rogueamoeba.com/audiohijack/) or Garage Band or really anything that works with Audio Unit plugins.

Then in your Python program simply write

```python
from pyaunetreceive import AUNetReceive

netrecv = AUNetReceive( '127.0.0.1', 52800 )
```
    
To configure a callack, write

```python
def callback( data ):
  print len(data)

netrecv.onrecv( callback )
```

Finally, to start listening, write

```python
netrecv.listen()
```

Data is in unsigned 16-bit format. The simplest way to process this data is with numpy, as in:

```python
import numpy

def callback( data ):
  
  interleaved = numpy.fromstring( data, dtype='int16' )

  left_channel = interleaved[::2]
  right_channel = interleaved[1::2]
```

From here, it's up to you!

## License

This projected is licensed under the terms of the MIT license.
