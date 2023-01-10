import dataclasses
import logging
import os
import struct
from typing import Tuple, Optional, DefaultDict, Dict
import numpy as np

import pyte
from pyte.graphics import FG_ANSI
from pyte.screens import Char

COLOR_DICT = dict((v, k) for k, v in FG_ANSI.items())


def ttyframes(f, tty2=True):
    frame_index = 0
    while True:
        if tty2:
            header = f.read(13)
        else:
            header = f.read(12)

        if not isinstance(header, bytes):
            raise IOError("File must be opened in binary mode.")

        if not header:
            return

        if tty2:
            sec, usec, length, channel = struct.unpack("<iiiB", header)
        else:
            sec, usec, length = struct.unpack("<iii", header)
            channel = 0

        if sec < 0 or usec < 0 or length < 0 or channel not in (0, 1, 2):
            raise IOError("Illegal header %s in %s" % ((sec, usec, length, channel), f))

        timestamp = sec + usec * 1e-6
        data = f.read(length)
        yield timestamp, channel, data  # Next execution resumes from this point
        frame_index += 1


def getfile(filename):
    if filename == "-":
        f = os.fdopen(os.dup(0), "rb")
        os.dup2(1, 0)
        return f
    elif os.path.splitext(filename)[1] in (".bz2", ".bzip2"):
        import bz2

        return bz2.BZ2File(filename)
    elif os.path.splitext(filename)[1] in (".gz", ".gzip"):
        import gzip

        return gzip.GzipFile(filename)
    else:
        return open(filename, "rb")


def decode_char(c: Char):
    # Two list as environment knowledge
    char_list = ['@', '.', '>', '<']
    color_list = [('white', 15), ('default', 7)]

    # Default values
    char_code = 32
    color_code = 0

    color_dict = dict(color_list)
    if c.data in char_list and c.fg in color_dict:
        color_code = color_dict[c.fg]
        char_code = ord(c.data)

    return char_code, color_code


# -------------------- THIS CLASS DESCRIBES THE BOTTOM LINE STATS -------------------- #
@dataclasses.dataclass
class Blstats(object):
    x: int = None
    y: int = None
    strength: int = None
    strength_percentage: int = None
    dexterity: int = None
    constitution: int = None
    intelligence: int = None
    wisdom: int = None
    charisma: int = None
    score: int = None
    hitpoints: int = None
    max_hitpoints: int = None
    depth: int = None
    gold: int = None
    energy: int = None
    max_energy: int = None
    armor_class: int = None
    monster_level: int = None
    experience_level: int = None
    experience_points: int = None
    time: int = None
    hunger_state: int = None
    carrying_capacity: int = None
    dungeon_number: int = None
    level_number: int = None


# ------------------------------------------------------------------------------------ #

# -------------------- THIS CLASS DESCRIBES THE FRAMES OF THE TTYREC RECORDINGS -------------------- #
class NethackFrame(object):
    def __init__(self, buffer: DefaultDict[int, Dict[int, Char]], width: int, height: int):
        self._buffer = buffer
        self._height = height
        self._width = width

    @property
    def agent_position(self) -> Optional[Tuple[int, int]]:
        for y in range(self._height):
            for x in range(self._width):
                char = self._buffer[y][x]
                if char.data == '@':
                    return y, x
        return None

    def _line(self, i):
        return ''.join([self._buffer[i][j].data for j in range(self._width)])

    @property
    def message(self) -> str:
        return self._line(0)

    @property
    def done(self):
        return "Really quit? [yn] (n)" in self._line(0)

    @property
    def chars(self):
        return [self._line(i) for i in range(1, self._height - 2)]

    @property
    def bottom_line_stats(self):
        return [self._line(self._height - 2), self._line(self._height - 1)]

    @property
    def window(self):
        return [self._line(i) for i in range(0, self._height)]

    def generate_observation(self):
        # These are the elements on which we base our observations
        chars = np.empty(1659, dtype=np.int16)
        colors = np.empty(1659, dtype=np.int16)

        i = 0
        for y in range(1, self._height - 2):  # 21
            for x in range(self._width - 1):  # 79
                char = self._buffer[y][x]
                chars[i], colors[i] = decode_char(char)
                i += 1

        # we flatten and return the observation
        obs = np.concatenate((chars, colors), axis=None)
        return obs

    def print_frame(self, grid, with_number: bool = True):
        if with_number:
            print(f"  | WINDOW WIDTH: {self._width}, WINDOW HEIGHT: {self._height}")
            print('--+' + ('-' * self._width))
        for i, row in enumerate(grid):
            if with_number:
                print(f'{i:2d}|', end='')
            print(f'{row}')
        print()

    def print_chars(self, with_number: bool = True):
        self.print_frame(self.chars, with_number)

    def print_window(self, with_number: bool = True):
        self.print_frame(self.window, with_number)


# -------------------------------------------------------------------------------------------------- #

# -------------------- THIS CLASS DESCRIBES A TTYREC READER -------------------- #
class TTYRecReader(object):
    def __init__(self, ttyrec_path, width=80, height=24):
        self._screen = pyte.HistoryScreen(width, height)  # width, height
        self._stream = pyte.ByteStream(self._screen)
        self._width = width
        self._height = height
        self._ttyrec_path = ttyrec_path

    def __iter__(self):
        try:
            with getfile(self._ttyrec_path) as f:
                for timestamp, channel, data in ttyframes(f):
                    if channel == 0:  # This will catch only the chars in frames
                        self._stream.feed(data)
                    frame = NethackFrame(self._screen.buffer, self._width, self._height)
                    if frame.agent_position is None:
                        continue
                    yield frame  # Next execution resumes from this point
        except Exception as e:
            logging.exception(e)
# ------------------------------------------------------------------------------ #
