/**
 * Piano Visualizer for StreamMUSE
 * Horizontal layout: keyboard on left, notes scroll right-to-left
 */

const PianoVisualizer = (function() {
    const canvas = document.getElementById('piano-canvas');
    const ctx = canvas.getContext('2d');
    
    const KEY_HEIGHT = 18;
    const WHITE_KEY_WIDTH = 80;
    const BLACK_KEY_WIDTH = 50;

    // Full 88-key piano: A0 (MIDI 21) → C8 (MIDI 108).
    // KEY_HEIGHT shrunk above so all 88 keys fit in a typical viewport
    // (52 white keys × 18px ≈ 936px tall).
    let visibleMinPitch = 21;   // A0
    let visibleMaxPitch = 108;  // C8
    
    let pixelsPerTick = 40;
    let currentTick = 0;
    // Horizontal visible range around NOW. Used for gridline rendering
    // and note cleanup horizon; wired from UI sliders via setVisibleTicks*.
    let visibleTicksAhead = 32;
    let visibleTicksBehind = 16;
    
    const pressedKeys = new Set();
    const noteHistory = [];
    const replacedNotes = [];
    // Open notes keyed by `${source}:${pitch}`. Each value points to a note
    // object inside noteHistory whose duration is still null (note_off
    // hasn't arrived yet on the wire).
    const openNotes = new Map();
    
    const COLORS = {
        background: '#f5f5f5',
        keyboardBg: '#ffffff',
        whiteKey: '#ffffff',
        whiteKeyPressed: '#bbdefb',
        whiteKeyBorder: '#ccc',
        blackKey: '#333333',
        blackKeyPressed: '#1565C0',
        
        user: '#2196F3',
        model: '#4CAF50',
        
        replaced: 'rgba(158, 158, 158, 0.4)',
        
        nowLine: '#E91E63',
        gridLine: 'rgba(0,0,0,0.08)',
        gridLineMajor: 'rgba(0,0,0,0.15)'
    };
    
    const KEYBOARD_WIDTH = WHITE_KEY_WIDTH + 20;
    let nowLineX = 0;
    
    function isBlackKey(pitch) {
        const note = pitch % 12;
        return [1, 3, 6, 8, 10].includes(note);
    }
    
    function getPitchY(pitch) {
        const reversedPitch = visibleMaxPitch - (pitch - visibleMinPitch);
        let whiteKeyCount = 0;
        for (let p = visibleMinPitch; p < reversedPitch; p++) {
            if (!isBlackKey(p)) whiteKeyCount++;
        }
        
        if (isBlackKey(pitch)) {
            let whiteKeysBeforeInReverse = 0;
            for (let p = visibleMinPitch; p < reversedPitch; p++) {
                if (!isBlackKey(p)) whiteKeysBeforeInReverse++;
            }
            return whiteKeysBeforeInReverse * KEY_HEIGHT - KEY_HEIGHT / 2;
        }
        return whiteKeyCount * KEY_HEIGHT;
    }
    
    function getNoteHeight(pitch) {
        return isBlackKey(pitch) ? KEY_HEIGHT - 4 : KEY_HEIGHT - 2;
    }
    
    function getTotalWhiteKeys() {
        let count = 0;
        for (let p = visibleMinPitch; p <= visibleMaxPitch; p++) {
            if (!isBlackKey(p)) count++;
        }
        return count;
    }
    
    function resizeCanvas() {
        const container = document.getElementById('piano-container');
        canvas.width = container.clientWidth;
        canvas.height = container.clientHeight;
        
        nowLineX = canvas.width * 0.4;
    }
    
    function getNoteColor(note) {
        if (note.source === 'user') return COLORS.user;
        return COLORS.model;
    }
    
    function tickToX(tick) {
        const tickDiff = tick - currentTick;
        return nowLineX + (tickDiff * pixelsPerTick);
    }
    
    function drawBackground() {
        ctx.fillStyle = COLORS.background;
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        const totalWhiteKeys = getTotalWhiteKeys();
        const pianoHeight = totalWhiteKeys * KEY_HEIGHT;
        const pianoStartY = (canvas.height - pianoHeight) / 2;
        
        ctx.fillStyle = 'rgba(0,0,0,0.02)';
        ctx.fillRect(KEYBOARD_WIDTH, pianoStartY, canvas.width - KEYBOARD_WIDTH, pianoHeight);
        
        for (let tick = currentTick - visibleTicksBehind; tick <= currentTick + visibleTicksAhead; tick++) {
            if (tick < 0) continue;
            const x = tickToX(tick);
            if (x < KEYBOARD_WIDTH || x > canvas.width) continue;
            
            const isMajor = tick % 4 === 0;
            ctx.strokeStyle = isMajor ? COLORS.gridLineMajor : COLORS.gridLine;
            ctx.lineWidth = isMajor ? 1.5 : 1;
            ctx.beginPath();
            ctx.moveTo(x, pianoStartY);
            ctx.lineTo(x, pianoStartY + pianoHeight);
            ctx.stroke();
        }
    }
    
    function drawNotes() {
        const totalWhiteKeys = getTotalWhiteKeys();
        const pianoHeight = totalWhiteKeys * KEY_HEIGHT;
        const pianoStartY = (canvas.height - pianoHeight) / 2;
        
        for (const note of replacedNotes) {
            if (note.pitch < visibleMinPitch || note.pitch > visibleMaxPitch) continue;
            
            const noteStartX = tickToX(note.startTick);
            const noteEndX = tickToX(note.startTick + note.duration);
            
            if (noteEndX < KEYBOARD_WIDTH || noteStartX > canvas.width) continue;
            
            const y = pianoStartY + getPitchY(note.pitch);
            const height = getNoteHeight(note.pitch);
            const width = Math.max(4, noteEndX - noteStartX);
            
            ctx.fillStyle = COLORS.replaced;
            ctx.fillRect(Math.max(KEYBOARD_WIDTH, noteStartX), y + 1, width, height);
            
            ctx.setLineDash([4, 4]);
            ctx.strokeStyle = 'rgba(158, 158, 158, 0.6)';
            ctx.lineWidth = 1;
            ctx.strokeRect(Math.max(KEYBOARD_WIDTH, noteStartX), y + 1, width, height);
            ctx.setLineDash([]);
        }
        
        for (const note of noteHistory) {
            if (note.pitch < visibleMinPitch || note.pitch > visibleMaxPitch) continue;

            // Our wire protocol sends note_on and note_off separately; a note
            // with duration=null is still open — grow it toward NOW.
            const effectiveDuration = note.duration == null
                ? Math.max(1, currentTick - note.startTick)
                : note.duration;
            const noteStartX = tickToX(note.startTick);
            const noteEndX = tickToX(note.startTick + effectiveDuration);

            if (noteEndX < KEYBOARD_WIDTH || noteStartX > canvas.width) continue;

            const y = pianoStartY + getPitchY(note.pitch);
            const height = getNoteHeight(note.pitch);
            const width = Math.max(4, noteEndX - noteStartX);

            const isPast = note.duration != null && note.startTick + note.duration < currentTick;
            let color = getNoteColor(note);
            
            if (isPast) {
                ctx.globalAlpha = 0.4;
            }
            
            ctx.fillStyle = color;
            ctx.fillRect(Math.max(KEYBOARD_WIDTH, noteStartX), y + 1, width, height);
            
            ctx.strokeStyle = 'rgba(0,0,0,0.2)';
            ctx.lineWidth = 1;
            ctx.strokeRect(Math.max(KEYBOARD_WIDTH, noteStartX), y + 1, width, height);
            
            ctx.globalAlpha = 1.0;
        }
        
        ctx.strokeStyle = COLORS.nowLine;
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.moveTo(nowLineX, pianoStartY - 10);
        ctx.lineTo(nowLineX, pianoStartY + pianoHeight + 10);
        ctx.stroke();
        
        ctx.fillStyle = COLORS.nowLine;
        ctx.font = 'bold 12px sans-serif';
        ctx.fillText('NOW', nowLineX - 18, pianoStartY - 15);
    }
    
    function drawPianoKeys() {
        const totalWhiteKeys = getTotalWhiteKeys();
        const pianoHeight = totalWhiteKeys * KEY_HEIGHT;
        const pianoStartY = (canvas.height - pianoHeight) / 2;
        
        ctx.fillStyle = COLORS.keyboardBg;
        ctx.fillRect(0, pianoStartY - 5, KEYBOARD_WIDTH, pianoHeight + 10);
        ctx.strokeStyle = '#ddd';
        ctx.lineWidth = 1;
        ctx.strokeRect(0, pianoStartY - 5, KEYBOARD_WIDTH, pianoHeight + 10);
        
        let whiteKeyIndex = 0;
        for (let pitch = visibleMaxPitch; pitch >= visibleMinPitch; pitch--) {
            if (!isBlackKey(pitch)) {
                const y = pianoStartY + whiteKeyIndex * KEY_HEIGHT;
                const isPressed = pressedKeys.has(pitch);
                
                ctx.fillStyle = isPressed ? COLORS.whiteKeyPressed : COLORS.whiteKey;
                ctx.fillRect(5, y, WHITE_KEY_WIDTH, KEY_HEIGHT - 1);
                
                ctx.strokeStyle = COLORS.whiteKeyBorder;
                ctx.lineWidth = 1;
                ctx.strokeRect(5, y, WHITE_KEY_WIDTH, KEY_HEIGHT - 1);
                
                if (pitch % 12 === 0) {
                    ctx.fillStyle = '#999';
                    ctx.font = '10px sans-serif';
                    ctx.fillText(`C${Math.floor(pitch / 12) - 1}`, WHITE_KEY_WIDTH - 20, y + KEY_HEIGHT / 2 + 3);
                }
                
                whiteKeyIndex++;
            }
        }
        
        whiteKeyIndex = 0;
        for (let pitch = visibleMaxPitch; pitch >= visibleMinPitch; pitch--) {
            if (!isBlackKey(pitch)) {
                whiteKeyIndex++;
            } else {
                const y = pianoStartY + whiteKeyIndex * KEY_HEIGHT - KEY_HEIGHT / 2;
                const isPressed = pressedKeys.has(pitch);
                
                ctx.fillStyle = isPressed ? COLORS.blackKeyPressed : COLORS.blackKey;
                ctx.fillRect(5, y, BLACK_KEY_WIDTH, KEY_HEIGHT - 2);
            }
        }
    }
    
    function render() {
        drawBackground();
        drawNotes();
        drawPianoKeys();
        requestAnimationFrame(render);
    }
    
    function cleanupOldNotes() {
        // Keep a small buffer past the visible-behind window so the edge
        // of the roll doesn't pop notes as it scrolls.
        const cutoffTick = currentTick - (visibleTicksBehind + 8);

        // Force-close open notes whose START is already past the cutoff.
        // Without this, an open note's right edge stays pinned to NOW
        // forever and the note never scrolls off the left edge — visible
        // as a long bar that won't disappear. This is the safety net for
        // the common case: a note_off that never arrived because the
        // model's response was replaced mid-flight.
        for (const [key, note] of openNotes) {
            if (note.startTick < cutoffTick) {
                note.duration = Math.max(1, currentTick - note.startTick);
                openNotes.delete(key);
            }
        }

        while (noteHistory.length > 0) {
            const n = noteHistory[0];
            // After the force-close above, only notes whose start is recent
            // can still be open. For those we keep them (still rendering).
            const endTick = n.duration == null ? currentTick : n.startTick + n.duration;
            if (endTick < cutoffTick) noteHistory.shift();
            else break;
        }
        while (replacedNotes.length > 0) {
            const n = replacedNotes[0];
            const endTick = n.duration == null ? currentTick : n.startTick + n.duration;
            if (endTick < cutoffTick) replacedNotes.shift();
            else break;
        }
    }

    function addNote(noteData) {
        // duration == 0 (or missing) means "unknown, grow until noteOff".
        // Positive values are honored verbatim.
        const hasKnownDuration = noteData.duration && noteData.duration > 0;
        const source = noteData.source || 'model';
        const startTick = noteData.tick;

        // Close-at-horizon for same-pitch retriggers. If a previous note for
        // this source+pitch is still open when a new note_on arrives, the
        // missing note_off is implicit — close the prior note 1 tick before
        // the new one starts. Mirrors the server-side converter policy
        // (domain/musical/converters.py).
        const key = `${source}:${noteData.pitch}`;
        const prior = openNotes.get(key);
        if (prior) {
            prior.duration = Math.max(1, startTick - prior.startTick);
            openNotes.delete(key);
        }

        const note = {
            pitch: noteData.pitch,
            startTick: startTick,
            duration: hasKnownDuration ? noteData.duration : null,
            source: source,
            id: noteData.id || null
        };
        noteHistory.push(note);
        if (!hasKnownDuration) {
            openNotes.set(key, note);
        }
    }

    function noteOff(pitch, tick, source) {
        const key = `${source || 'model'}:${pitch}`;
        const open = openNotes.get(key);
        if (!open) return;
        open.duration = Math.max(1, tick - open.startTick);
        openNotes.delete(key);
    }
    
    function replaceNotes(startTick, newNotes) {
        const toReplace = [];
        const toKeep = [];
        
        for (const note of noteHistory) {
            if (note.source !== 'user' && note.startTick >= startTick) {
                toReplace.push({...note, replacedAt: currentTick});
            } else {
                toKeep.push(note);
            }
        }
        
        replacedNotes.push(...toReplace);
        noteHistory.length = 0;
        noteHistory.push(...toKeep);
        
        for (const noteData of newNotes) {
            addNote(noteData);
        }
    }
    
    function pressKey(pitch) {
        pressedKeys.add(pitch);
    }
    
    function releaseKey(pitch) {
        pressedKeys.delete(pitch);
    }
    
    function setCurrentTick(tick) {
        currentTick = tick;
        cleanupOldNotes();
    }
    
    function setPixelsPerTick(value) {
        pixelsPerTick = value;
    }
    
    function setVisibleTicksAhead(value) {
        visibleTicksAhead = Math.max(1, parseInt(value) || 32);
    }
    function setVisibleTicksBehind(value) {
        visibleTicksBehind = Math.max(1, parseInt(value) || 16);
    }
    
    function clearNotes() {
        noteHistory.length = 0;
        replacedNotes.length = 0;
        pressedKeys.clear();
        openNotes.clear();
    }
    
    window.addEventListener('resize', resizeCanvas);
    resizeCanvas();
    render();
    
    return {
        addNote,
        noteOff,
        replaceNotes,
        pressKey,
        releaseKey,
        setCurrentTick,
        setPixelsPerTick,
        setVisibleTicksAhead,
        setVisibleTicksBehind,
        clearNotes
    };
})();
