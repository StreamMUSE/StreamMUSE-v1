# StreamMUSE Client Pipeline

This document describes the multi-threaded architecture of the StreamMUSE client, detailing what occurs in each thread in chronological/loop order.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CLIENT                                          │
│                                                                              │
│  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐       │
│  │  Input Thread    │    │  Main Thread     │    │ Inference Worker │       │
│  │  (optional)      │    │  (Tick Loop)     │    │    Thread        │       │
│  └────────┬─────────┘    └────────┬─────────┘    └────────┬─────────┘       │
│           │                       │                       │                  │
│           │   event_queue         │  request_queue        │                  │
│           └──────────────────────►│◄──────────────────────┘                  │
│                                   │  response_queue                          │
│                                   │◄──────────────────────┐                  │
│                                   │                       │                  │
└───────────────────────────────────┼───────────────────────┼──────────────────┘
                                    │                       │
                                    │                       │
                                    ▼                       ▼
                            ┌───────────────┐       ┌───────────────┐
                            │ Audio Output  │       │    Server     │
                            │ (MIDI Device) │       │  (Inference)  │
                            └───────────────┘       └───────────────┘
```

## Threads

### 1. Input Thread (Optional)

Handles user input collection. Runs in background, puts events into `event_queue`.

**Input Sources (one of):**
- MIDI Device (`read_midi_input`)
- Computer Keyboard (`read_keyboard_input`)
- MIDI File (`read_midi_file_input`)

**Loop:**
```
while running:
    1. Wait for input event (note_on/note_off)
    2. Create event dict: {type, pitch, velocity}
    3. Put event into event_queue
    4. IMMEDIATELY play audio via audio_handler (if provided)
       - This ensures zero-latency audio feedback for user input
       - No need to wait for tick loop to process the event
```

---

### 2. Inference Worker Thread

Handles HTTP communication with the server. Runs in background, blocking on request queue.

**Loop:**
```
while True:
    1. BLOCK: Wait for request from inference_request_queue
       - If None received, exit thread
    
    2. PREPARE REQUEST:
       - Add client_request_send_time timestamp
       - request_data = {
           melody_notes: [...],
           generation_start_tick: int,
           generation_length_frames: int
         }
    
    3. SEND: POST request to server_url
       - Measure round_trip_time
    
    4. RECEIVE: Parse JSON response
       - response = {
           accompaniment: [{pitch, tick, duration, program}, ...],
           timings: {request_arrival_time, response_output_time, ...}
         }
    
    5. PUT: (response, round_trip_time, request_data) into inference_response_queue
```

---

### 3. Main Thread (Tick Loop)

The heart of the client. Executes once per tick at tempo-defined intervals.

**Timing:**
```
seconds_per_tick = (60.0 / tempo) / ticks_per_beat
```

**State Variables:**
- `tick_count`: Current tick number
- `playback_schedule`: Dict mapping tick -> list of scheduled events
- `notes_for_next_request`: Buffer of user notes to send in next request

**Loop (each tick):**

```
tick_count += 1

# ═══════════════════════════════════════════════════════════════
# STEP 1: Process User Input (from event_queue)
# ═══════════════════════════════════════════════════════════════
# NOTE: Audio playback for user notes now happens in the Input Thread
# for zero-latency feedback. This step only handles quantization and logging.
while event_queue not empty:
    event = event_queue.get()
    
    if event.type == "note_on":
        # Quantize note to current tick
        quantized_note = {
            pitch: event.pitch,
            tick: tick_count - 1,
            duration: NOTE_DURATION_TICKS (default 4)
        }
        
        # Buffer for next server request
        notes_for_next_request.append(quantized_note)
        
        # Log for MIDI file output
        midi_file_handler.add_user_note(quantized_note)
        
        # Audio already played in Input Thread
    
    elif event.type == "note_off":
        # Audio already handled in Input Thread
        pass

# ═══════════════════════════════════════════════════════════════
# STEP 2: Process Inference Responses (from inference_response_queue)
# ═══════════════════════════════════════════════════════════════
while inference_response_queue not empty:
    (response_data, round_trip_time, request_data) = queue.get()
    
    if response_data:
        # Calculate timing metrics
        timings = response_data.timings
        timings.round_trip_time = round_trip_time
        timings.server_processing_duration = 
            timings.response_output_time - timings.request_arrival_time
        timings.total_network_latency = 
            round_trip_time - server_processing_duration
        
        # Get newly generated accompaniment notes
        newly_generated_notes = response_data.accompaniment
        
        # Assign backup_level based on tick offset from generation_start_tick
        for note in newly_generated_notes:
            note.backup_level = note.tick - request_data.generation_start_tick
        
        # ─────────────────────────────────────────────────────────
        # BACKUP MECHANISM: Clear stale notes from previous generation
        # ─────────────────────────────────────────────────────────
        if newly_generated_notes:
            first_new_note_tick = min(note.tick for note in newly_generated_notes)
            
            # Remove model-generated events at/after first_new_note_tick
            for tick in playback_schedule where tick >= first_new_note_tick:
                playback_schedule[tick] = [
                    event for event in playback_schedule[tick]
                    if event.source != "model"  # Keep user events
                ]
        
        # ─────────────────────────────────────────────────────────
        # Schedule new notes for playback
        # ─────────────────────────────────────────────────────────
        for note in newly_generated_notes:
            if note.tick >= tick_count:  # Only future notes
                playback_schedule[note.tick].append({
                    ...note,
                    source: "model"
                })

# ═══════════════════════════════════════════════════════════════
# STEP 3: Trigger New Inference Request (if trigger tick)
# ═══════════════════════════════════════════════════════════════
is_trigger_tick = (tick_count % generation_interval_ticks) == 0

if is_trigger_tick:
    next_interval_start_tick = tick_count + 1
    
    request_data = {
        melody_notes: notes_for_next_request,
        generation_start_tick: next_interval_start_tick,
        generation_length_frames: generation_length_per_request
    }
    
    inference_request_queue.put((request_data, request_data.copy()))
    notes_for_next_request = []  # Clear buffer

# ═══════════════════════════════════════════════════════════════
# STEP 4: Play Scheduled Notes (for current tick)
# ═══════════════════════════════════════════════════════════════
scheduled_events = playback_schedule.pop(tick_count, [])

for event in scheduled_events:
    if event.type == "note_off":
        audio_output.off(event.pitch)
    else:  # note_on
        audio_output.on(
            event.pitch, 
            accompaniment_velocity, 
            channel=ACCOMPANIMENT_CHANNEL
        )
        midi_file_handler.add_model_note(event)
        
        # Schedule corresponding note_off
        note_off_tick = tick_count + event.duration
        playback_schedule[note_off_tick].append({
            ...event,
            type: "note_off",
            source: "model"
        })

# ═══════════════════════════════════════════════════════════════
# STEP 5: Metronome
# ═══════════════════════════════════════════════════════════════
if metronome_enabled:
    is_beat_tick = (tick_count % ticks_per_beat) == 0
    if is_beat_tick:
        beat_in_bar = (tick_count % ticks_per_bar) // ticks_per_beat
        if beat_in_bar == 0:
            audio_output.metro_first()   # Accent on downbeat
        else:
            audio_output.metro_other()

# ═══════════════════════════════════════════════════════════════
# STEP 6: Update Display
# ═══════════════════════════════════════════════════════════════
output_handler.update_and_display(
    tick_count,
    music_info,       # bar, beat, timing stats
    user_notes,       # notes user played this tick
    model_notes,      # notes model played this tick  
    pending_notes     # notes buffered for next request
)

# ═══════════════════════════════════════════════════════════════
# STEP 7: Wait for Next Tick
# ═══════════════════════════════════════════════════════════════
time.sleep(seconds_per_tick)
```

---

## Data Flow Timeline

```
Tick 0          Tick 1          Tick 2          Tick 3          Tick 4
  │               │               │               │               │
  │ User plays    │               │               │               │
  │ note C4       │               │               │               │
  │   │           │               │               │               │
  │   ▼           │               │               │               │
  │ Quantize to   │               │               │               │
  │ tick=-1       │               │               │               │
  │   │           │               │               │               │
  │   ▼           │               │               │               │
  │ Buffer note   │               │               │               │
  │               │               │               │               │
  │ Trigger tick! │               │               │               │
  │   │           │               │               │               │
  │   ▼           │               │               │               │
  │ Send request ─┼──────────────►│ Server        │               │
  │ gen_start=1   │               │ processing    │               │
  │               │               │   │           │               │
  │               │               │   ▼           │               │
  │               │               │ Response ─────┼──────────────►│
  │               │               │ notes for     │               │
  │               │               │ ticks 1,2,3   │               │
  │               │               │               │               │
  │               │ Play note     │ Play note     │ Play note     │
  │               │ tick=1        │ tick=2        │ tick=3        │
```

---

## Backup Mechanism Explained

When a new inference response arrives, it may contain notes for ticks that already have scheduled notes from a previous response. The backup mechanism:

1. **Identifies overlap**: Find the first tick in the new response
2. **Clears stale notes**: Remove all model-generated events at or after that tick
3. **Preserves user events**: User note_off events are never removed
4. **Schedules new notes**: Add all notes from the new response

**backup_level** indicates how "late" the response arrived:
- `backup_level = 0`: Note plays at the exact tick it was intended for
- `backup_level = 1`: Note plays 1 tick after intended (response arrived late)
- `backup_level = 2+`: Response was significantly delayed

---

## Queue Summary

| Queue | Producer | Consumer | Contents |
|-------|----------|----------|----------|
| `event_queue` | Input Thread | Main Thread | User input events |
| `inference_request_queue` | Main Thread | Inference Worker | Server requests |
| `inference_response_queue` | Inference Worker | Main Thread | Server responses |

---

## Listening Mode (Optional)

When `listening_duration_ticks > 0`, the client operates in two phases:

**Phase 1: Listening Period (ticks 0 to listening_duration_ticks)**
- Collect user notes but don't send inference requests
- At end, spawn background thread for key detection

**Phase 2: Grace Period (8 ticks)**
- Wait for key detection to complete
- Inject detected key's prompt to server

**Phase 3: Normal Operation**
- Resume standard tick loop with inference requests
