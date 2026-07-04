/**
 * Action registry tests (2.4) — one source of truth for keys, palette
 * and cheat sheet.
 */
import { describe, it, expect } from 'vitest';
import { EDITOR_ACTIONS, matchAction, fuzzyScore, getActionById } from './editorActions';

const evt = (code, { mod = false, shift = false, alt = false } = {}) => ({
  code, ctrlKey: mod, metaKey: false, shiftKey: shift, altKey: alt,
});

describe('editor action registry', () => {
  it('has unique ids', () => {
    const ids = EDITOR_ACTIONS.map((a) => a.id);
    expect(new Set(ids).size).toBe(ids.length);
  });

  it('every action has label, category, kbd and a run function', () => {
    for (const a of EDITOR_ACTIONS) {
      expect(a.label, a.id).toBeTruthy();
      expect(a.category, a.id).toBeTruthy();
      expect(a.kbd, a.id).toBeTruthy();
      expect(typeof a.run, a.id).toBe('function');
    }
  });

  it('no two dispatchable actions claim the same key combo', () => {
    const combos = new Map();
    for (const a of EDITOR_ACTIONS) {
      if (a.holdRepeat) continue;
      for (const code of (a.codes || [a.code])) {
        const key = `${code}|${!!a.mod}|${!!a.shift}|${!!a.alt}`;
        expect(combos.has(key), `${a.id} collides with ${combos.get(key)} on ${key}`).toBe(false);
        combos.set(key, a.id);
      }
    }
  });

  it('matchAction resolves modifiers exactly', () => {
    expect(matchAction(evt('KeyZ', { mod: true }))?.id).toBe('undo');
    expect(matchAction(evt('KeyZ', { mod: true, shift: true }))?.id).toBe('redo');
    expect(matchAction(evt('KeyZ', { shift: true }))?.id).toBe('zoom-to-fit');
    expect(matchAction(evt('KeyZ'))).toBeNull();
    expect(matchAction(evt('KeyK', { mod: true }))?.id).toBe('command-palette');
    expect(matchAction(evt('Slash', { shift: true }))?.id).toBe('shortcut-help');
    expect(matchAction(evt('KeyL'))?.id).toBe('shuttle-forward');
    expect(matchAction(evt('KeyL', { shift: true }))?.id).toBe('loop-toggle');
    expect(matchAction(evt('Delete'))?.id).toBe('delete-selection');
    expect(matchAction(evt('Backspace'))?.id).toBe('delete-selection');
    // holdRepeat arrows are the hook's job, never dispatched here
    expect(matchAction(evt('ArrowLeft'))).toBeNull();
  });

  it('getActionById feeds tooltip kbd chips', () => {
    expect(getActionById('undo').kbd).toBe('⌘Z');
    expect(getActionById('nope')).toBeNull();
  });
});

describe('fuzzyScore', () => {
  it('matches subsequences and ranks word starts higher', () => {
    expect(fuzzyScore('spl', 'Split clip at playhead')).toBeGreaterThan(0);
    expect(fuzzyScore('xyz', 'Split clip at playhead')).toBe(-1);
    const wordStart = fuzzyScore('play', 'Play / pause');
    const scattered = fuzzyScore('play', 'Split at playhead');
    expect(wordStart).toBeGreaterThan(0);
    expect(scattered).toBeGreaterThan(0);
  });

  it('empty query matches everything with score 0', () => {
    expect(fuzzyScore('', 'anything')).toBe(0);
  });
});
