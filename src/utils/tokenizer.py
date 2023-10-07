from mido import MidiFile, MidiTrack, Message, MetaMessage
from src.utils.vocab import *
from torch import LongTensor, Tensor, multiprocessing
multiprocessing.set_sharing_strategy('file_system')
import os
import sys
sys.path.append(os.getcwd())
class Tokenizer:
    def midi2tensor(self, fname: str):
        mid        = MidiFile(fname)
        delta_time = 0          # time between important midi messages
        event_list = []         # list of events in vocab
        encode_list = []        # list of indices in vocab
        pedal_events = {}       # dict to handle pedal events
        pedal_flag = False      # flag to handle pedal events
        
        for track in mid.tracks:
            for msg in track:
                delta_time += msg.time
                if msg.is_meta:
                    continue
                vel = 0   # velocity
                if msg.type == "note_on":
                    idx = msg.note + 1
                    vel = velocity_to_bin(msg.velocity)
                elif msg.type == "note_off":
                    note = msg.note
                    if pedal_flag:
                        if note not in pedal_events:
                            pedal_events[note] = 0
                        pedal_events[note] += 1
                        continue
                    else:
                        idx = NOTE_ON + note + 1
                elif msg.type == "control_change":
                    if msg.control == 64:
                        if msg.value >= 64:
                            pedal_flag = True
                        elif pedal_events:
                            pedal_flag = False
                            time_to_events(delta_time, event_list=event_list, index_list=encode_list)
                            delta_time = 0
                            for note in pedal_events:
                                idx = NOTE_ON + note + 1
                                for i in range(pedal_events[note]):
                                    event_list.append(vocab[idx])
                                    encode_list.append(idx)
                            pedal_events = {}
                    continue
                else:
                    continue
                time_to_events(delta_time, event_list=event_list, index_list=encode_list)
                delta_time = 0
                if msg.type == "note_on":
                    event_list.append(vocab[NOTE_ON + NOTE_OFF + TIME_SHIFT + vel + 1])
                    encode_list.append(NOTE_ON + NOTE_OFF + TIME_SHIFT + vel + 1)
                event_list.append(vocab[idx])
                encode_list.append(idx)
        return LongTensor(encode_list), event_list

    def tensor2midi(self,
                    encode_tensor:Tensor, 
                    save_dir:str, 
                    tempo=512820):

        mid        = MidiFile()
        meta_track = MidiTrack()
        track      = MidiTrack()
        time_sig   = MetaMessage("time_signature")
        time_sig   = time_sig.copy(numerator=4, denominator=4, time=0)    
        key_sig    = MetaMessage("key_signature", time=0)     
        set_tempo  = MetaMessage("set_tempo")
        set_tempo  = set_tempo.copy(tempo=tempo, time=0)       
        end        = MetaMessage("end_of_track").copy(time=0)       
        program    = Message("program_change", channel=0, program=0, time=0)
        cc         = Message("control_change", time=0)
        
        meta_track.append(MetaMessage("track_name").copy(name=save_dir, time=0))
        meta_track.append(MetaMessage("smpte_offset"))
        meta_track.append(time_sig)
        meta_track.append(key_sig)
        meta_track.append(set_tempo)
        meta_track.append(end)
        track.append(program)
        track.append(cc)
        delta_time = 0
        vel       = 0

        for idx in encode_tensor:
            idx = idx.item()
            idx = idx - 1
            if 0 <= idx < NOTE_ON + NOTE_OFF:
                # note on event
                if 0 <= idx < NOTE_ON:
                    note = idx
                    t = "note_on"
                    v = vel  
                else:
                    note = idx - NOTE_ON
                    t = "note_off"
                    v = 127
                msg = Message(t)
                msg = msg.copy(note=note, velocity=v, time=delta_time)
                track.append(msg)
                delta_time = 0
                vel = 0

            elif NOTE_ON + NOTE_OFF <= idx < NOTE_ON + NOTE_OFF + TIME_SHIFT:
                cut_time = idx - (NOTE_ON + NOTE_OFF - 1)
                delta_time += cut_time * DIV

            elif NOTE_ON + NOTE_OFF + TIME_SHIFT <= idx < TOTAL_MIDI_EVENTS:
                vel = bin_to_velocity(idx - (NOTE_ON + NOTE_OFF + TIME_SHIFT))

        end = MetaMessage("end_of_track").copy(time=0)
        track.append(end)

        mid.tracks.append(meta_track)
        mid.tracks.append(track)
        return mid

if __name__ == '__main__':
    tokenizer = Tokenizer()
    tokens = tokenizer.midi2tensor('./src/utils/piano.mid')
    decode = tokenizer.tensor2midi(tokens[0], './src/utils/test.mid')
    print(tokens[1][:100])
    print(decode)