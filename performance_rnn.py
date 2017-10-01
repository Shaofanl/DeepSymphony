#####################################################################
# from magenta/magenta/models/performance_rnn/performance_encoder_decoder
#####################################################################
EVENT_RANGES = [
    (PerformanceEvent.NOTE_ON,
     performance_lib.MIN_MIDI_PITCH, performance_lib.MAX_MIDI_PITCH),
    (PerformanceEvent.NOTE_OFF,
     performance_lib.MIN_MIDI_PITCH, performance_lib.MAX_MIDI_PITCH),
    (PerformanceEvent.TIME_SHIFT, 1, performance_lib.MAX_SHIFT_STEPS),
]

class PerformanceOneHotEncoding(encoder_decoder.OneHotEncoding):
  """One-hot encoding for performance events."""

  def __init__(self, num_velocity_bins=0):
    if num_velocity_bins > 0:
      self._event_ranges = EVENT_RANGES + [
          (PerformanceEvent.VELOCITY, 1, num_velocity_bins)]
    else:
      self._event_ranges = EVENT_RANGES

  @property
  def num_classes(self):
    return sum(max_value - min_value + 1
               for event_type, min_value, max_value in self._event_ranges)

  @property
  def default_event(self):
    return PerformanceEvent(
        event_type=PerformanceEvent.TIME_SHIFT,
        event_value=performance_lib.MAX_SHIFT_STEPS)

  def encode_event(self, event):
    offset = 0
    for event_type, min_value, max_value in self._event_ranges:
      if event.event_type == event_type:
        return offset + event.event_value - min_value
      offset += max_value - min_value + 1

    raise ValueError('Unknown event type: %s' % event.event_type)

  def decode_event(self, index):
    offset = 0
    for event_type, min_value, max_value in self._event_ranges:
      if offset <= index <= offset + max_value - min_value:
        return PerformanceEvent(
            event_type=event_type, event_value=min_value + index - offset)
      offset += max_value - min_value + 1

    raise ValueError('Unknown event index: %s' % index)


class NoteDensityOneHotEncoding(encoder_decoder.OneHotEncoding):
  """One-hot encoding for performance note density events.

  Encodes by quantizing note density events. When decoding, always decodes to
  the minimum value for each bin. The first bin starts at zero note density.
  """

  def __init__(self, density_bin_ranges):
    """Initialize a NoteDensityOneHotEncoding.

    Args:
      density_bin_ranges: List of note density (notes per second) bin boundaries
          to use when quantizing. The number of bins will be one larger than the
          list length.
    """
    self._density_bin_ranges = density_bin_ranges

  @property
  def num_classes(self):
    return len(self._density_bin_ranges) + 1

  @property
  def default_event(self):
    return 0.0

  def encode_event(self, event):
    for idx, density in enumerate(self._density_bin_ranges):
      if event < density:
        return idx
    return len(self._density_bin_ranges)

  def decode_event(self, index):
    if index == 0:
      return 0.0
    else:
      return self._density_bin_ranges[index - 1]


class PitchHistogramEncoderDecoder(encoder_decoder.EventSequenceEncoderDecoder):
  """An encoder/decoder for pitch class histogram sequences.

  This class has no label encoding and is only a trivial input encoder that
  merely uses each histogram as the input vector.
  """

  @property
  def input_size(self):
    return NOTES_PER_OCTAVE

  @property
  def num_classes(self):
    raise NotImplementedError

  @property
  def default_event_label(self):
    raise NotImplementedError

  def events_to_input(self, events, position):
    return events[position]

  def events_to_label(self, events, position):
    raise NotImplementedError

  def class_index_to_event(self, class_index, events):
    raise NotImplementedError


#####################################################################
# from magenta/magenta/models/performance_rnn/performance_lib.py_
#####################################################################
  @staticmethod
  def _from_quantized_sequence(quantized_sequence, start_step=0,
                               num_velocity_bins=0):
    """Populate self with events from the given quantized NoteSequence object.

    Within a step, new pitches are started with NOTE_ON and existing pitches are
    ended with NOTE_OFF. TIME_SHIFT shifts the current step forward in time.
    VELOCITY changes the current velocity value that will be applied to all
    NOTE_ON events.

    Args:
      quantized_sequence: A quantized NoteSequence instance.
      start_step: Start converting the sequence at this time step.
      num_velocity_bins: Number of velocity bins to use. If 0, velocity events
          will not be included at all.

    Returns:
      A list of events.
    """
    notes = [note for note in quantized_sequence.notes
             if not note.is_drum and note.quantized_start_step >= start_step]
    sorted_notes = sorted(notes, key=lambda note: note.start_time)

    # Sort all note start and end events.
    onsets = [(note.quantized_start_step, idx, False)
              for idx, note in enumerate(sorted_notes)]
    offsets = [(note.quantized_end_step, idx, True)
               for idx, note in enumerate(sorted_notes)]
    note_events = sorted(onsets + offsets)

    if num_velocity_bins:
      velocity_bin_size = int(math.ceil(
          (MAX_MIDI_VELOCITY - MIN_MIDI_VELOCITY + 1) / num_velocity_bins))
      velocity_to_bin = (
          lambda v: (v - MIN_MIDI_VELOCITY) // velocity_bin_size + 1)

    current_step = start_step
    current_velocity_bin = 0
    performance_events = []

    for step, idx, is_offset in note_events:
      if step > current_step:
        # Shift time forward from the current step to this event.
        while step > current_step + MAX_SHIFT_STEPS:
          # We need to move further than the maximum shift size.
          performance_events.append(
              PerformanceEvent(event_type=PerformanceEvent.TIME_SHIFT,
                               event_value=MAX_SHIFT_STEPS))
          current_step += MAX_SHIFT_STEPS
        performance_events.append(
            PerformanceEvent(event_type=PerformanceEvent.TIME_SHIFT,
                             event_value=int(step - current_step)))
        current_step = step

      # If we're using velocity and this note's velocity is different from the
      # current velocity, change the current velocity.
      if num_velocity_bins:
        velocity_bin = velocity_to_bin(sorted_notes[idx].velocity)
        if not is_offset and velocity_bin != current_velocity_bin:
          current_velocity_bin = velocity_bin
          performance_events.append(
              PerformanceEvent(event_type=PerformanceEvent.VELOCITY,
                               event_value=current_velocity_bin))

      # Add a performance event for this note on/off.
      event_type = (PerformanceEvent.NOTE_OFF if is_offset
                    else PerformanceEvent.NOTE_ON)
      performance_events.append(
          PerformanceEvent(event_type=event_type,
                           event_value=sorted_notes[idx].pitch))

    return performance_events
