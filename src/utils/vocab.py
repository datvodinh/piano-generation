"""
    vocab[       0] = '<PAD>'
    vocab[  1..128] = note_on
    vocab[129..256] = note_off
    vocab[257..381] = time_shift
    vocab[382..413] = velocity
    vocab[414..415] = '<SOS>', '<END>'
"""
NOTE_ON = 128
NOTE_OFF = 128
NOTE_EVENTS = 256
TIME_SHIFT = 125
VELOCITY = 32
LTH = 1000  # max milliseconds; LTH ms = 125 TIME_SHIFT
DIV = LTH // TIME_SHIFT  # 1 time_shift = DIV milliseconds
BIN_STEP = 128 // VELOCITY  # number of velocities per bin
TOTAL_MIDI_EVENTS = NOTE_ON + NOTE_OFF + TIME_SHIFT + VELOCITY

note_on_vocab = [f"note_on_{i}" for i in range(NOTE_ON)]
note_off_vocab = [f"note_off_{i}" for i in range(NOTE_OFF)]
time_shift_vocab = [f"time_shift_{i}" for i in range(TIME_SHIFT)]
velocity_vocab = [f"set_velocity_{i}" for i in range(VELOCITY)]

vocab = ['<PAD>'] + note_on_vocab + note_off_vocab + time_shift_vocab + velocity_vocab + ['<SOS>', '<EOS>']
vocab_size = len(vocab)

pad_token = vocab.index("<PAD>")
start_token = vocab.index("<SOS>")
end_token = vocab.index("<EOS>")

def events_to_indices(event_list, _vocab=None):
    if _vocab is None:
        _vocab = vocab
    index_list = []
    for event in event_list:
        index_list.append(_vocab.index(event))
    return index_list

def indices_to_events(index_list, _vocab=None):
    if _vocab is None:
        _vocab = vocab
    event_list = []
    for idx in index_list:
        event_list.append(_vocab[idx])
    return event_list

def velocity_to_bin(velocity, step=BIN_STEP):
    _bin = velocity // step
    return _bin

def bin_to_velocity(_bin, step=BIN_STEP):
    return int(_bin * step)

def time_to_events(delta_time, event_list=None, index_list=None, _vocab=None):
    if _vocab is None:
        _vocab = vocab
    time = time_cutter(delta_time)
    for i in time:
        idx = NOTE_ON + NOTE_OFF + i
        if event_list is not None:
            event_list.append(_vocab[idx])
        if index_list is not None:
            index_list.append(idx)
    return

def time_cutter(time, lth=LTH, div=DIV):
    if lth % div != 0:
        raise ValueError("lth must be divisible by div")
    time_shifts = []

    for i in range(time // lth):
        time_shifts.append(round(lth / div + 1e-4))
    leftover_time_shift = round((time % lth) / div + 1e-4)
    time_shifts.append(leftover_time_shift) if leftover_time_shift > 0 else None
    return time_shifts