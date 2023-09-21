# JSON Stream Parser

JSON Stream Parser is an opportunistic parser designed for processing incomplete JSON strings in real-time from streams.

This can be very useful for parsing JSON strings from network streams, such as websockets and don't want to wait for
the entire JSON string to be received before acting on it.

A good use-case is for parsing OpenAI's GPT response with streaming completion enabled or even with functions enabled.
(This is actually a real use-case! see how [openai-streaming](https://github.com/AlmogBaku/openai-streaming) use this
package just for that)

## Installation

Install the package using pip:

```bash
pip install json-stream
```

## Features

* **Real-time Parsing**: Process incomplete JSON strings as they are streamed.
* **Flexibility**: Built on a generator-based architecture for ease of integration.
* **Partial Object Handling**: Yields partially parsed JSON objects to be acted upon immediately.

## Usage

### Basic Usage

```python
from json_stream import loads

# Define a stream of JSON data
def stream_json():
    json_string = '{"field1": "value1", "field2": 42}'
    chunk_size = 4
    for i in range(0, len(json_string), chunk_size):
        yield json_string[i:i + chunk_size]

# Create a stream generator
mystream = stream_json()

# Use the 'loads()' function to parse the stream
ret = loads(mystream)

# Iterate through the generator to get parsed JSON objects
for s in ret:
    print(s)  # Output will be tuples of (ParseState, JSON object)
```

The `loads()` function returns a generator that yields tuples of `(ParseState, JSON object)`.

### Using with Generator receivers

Utilize the parser with a generator receiver for additional control:

```python
from json_stream import loads

# Initialize the parser generator
parser = loads()
next(parser)

# Stream and send chunks to the parser
for chunk in stream_json():
    recv = parser.send(chunk)
    print(recv)  # Output will be tuples of (ParseState, JSON object)
```

## How it Works

The parser internally maintains a stack of opening symbols like brackets, braces, and quotes. Upon each new data chunk,
it attempts to match these with closing symbols to parse available JSON objects.

## License

This project is licensed under the terms of the MIT license.
