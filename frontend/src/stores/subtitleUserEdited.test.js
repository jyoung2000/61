import { describe, it, expect, beforeEach } from 'vitest';
import useTimelineStore from './timelineStore.js';

// Regression: programmatic transcript/translation → timeline reconciliation
// must NOT flag the subtitle track as user-edited, otherwise the stale-language
// rebuild is blocked and un-matched cues stay in the source language ("subtitle
// elements still in Japanese after translation"). Genuine user edits still flag.

describe('subtitlesUserEdited flag — user vs programmatic subtitle changes', () => {
  beforeEach(() => {
    useTimelineStore.setState({
      items: [
        { id: 's1', trackId: 't1', type: 'subtitle', start: 0, end: 1, subtitleText: 'こんにちは' },
        { id: 's2', trackId: 't1', type: 'subtitle', start: 1, end: 2, subtitleText: 'さようなら' },
      ],
      subtitlesUserEdited: false,
    });
  });

  it('programmatic update (markUserEdit:false) does not set the flag', () => {
    useTimelineStore.getState().updateItem('s1', { subtitleText: 'Hello' }, { markUserEdit: false });
    const st = useTimelineStore.getState();
    expect(st.subtitlesUserEdited).toBe(false);
    expect(st.items.find((i) => i.id === 's1').subtitleText).toBe('Hello');
  });

  it('genuine user update sets the flag', () => {
    useTimelineStore.getState().updateItem('s1', { subtitleText: 'Hello' });
    expect(useTimelineStore.getState().subtitlesUserEdited).toBe(true);
  });

  it('programmatic remove (markUserEdit:false) does not set the flag', () => {
    useTimelineStore.getState().removeItem('s2', { markUserEdit: false });
    const st = useTimelineStore.getState();
    expect(st.subtitlesUserEdited).toBe(false);
    expect(st.items.some((i) => i.id === 's2')).toBe(false);
  });

  it('genuine user remove sets the flag', () => {
    useTimelineStore.getState().removeItem('s2');
    expect(useTimelineStore.getState().subtitlesUserEdited).toBe(true);
  });
});
