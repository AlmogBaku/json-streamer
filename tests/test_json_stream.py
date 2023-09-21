import unittest
from json_stream import loads, ParseState


def stream_json():
    json_string = '{"field1": "value1", "field2": 42}'
    chunk_size = 4
    for i in range(0, len(json_string), chunk_size):
        yield json_string[i:i + chunk_size]


class TestStreamJson(unittest.TestCase):

    def test_stream_json(self):
        mystream = stream_json()
        ret = loads(mystream)

        expected_states = [
            (ParseState.PARTIAL, {'field1': ''}),
            (ParseState.PARTIAL, {'field1': 'valu'}),
            (ParseState.PARTIAL, {'field1': 'value1', 'field2': 4}),
            (ParseState.COMPLETE, {'field1': 'value1', 'field2': 42})
        ]

        for i, (state, parsed_dict) in enumerate(ret):
            expected_state, expected_dict = expected_states[i]
            self.assertEqual(state, expected_state)
            self.assertEqual(parsed_dict, expected_dict)


if __name__ == '__main__':
    unittest.main()
