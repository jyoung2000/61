// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { dispositionName, saveTextAs } from './downloadFile';

const _res = (dispo) => ({
  headers: { get: (k) => (k.toLowerCase() === 'content-disposition' ? dispo : null) },
});

describe('dispositionName', () => {
  it('reads the quoted filename the backend chose', () => {
    expect(dispositionName(_res('attachment; filename="My Talk_translated.srt"')))
      .toBe('My Talk_translated.srt');
  });

  it('reads an unquoted filename', () => {
    expect(dispositionName(_res('attachment; filename=clip.vtt'))).toBe('clip.vtt');
  });

  it('decodes the RFC 5987 form', () => {
    expect(dispositionName(_res("attachment; filename*=UTF-8''Cami%C3%B3n.srt")))
      .toBe('Camión.srt');
  });

  it('keeps a literal percent that is not an escape', () => {
    expect(dispositionName(_res('attachment; filename="100% done.srt"')))
      .toBe('100% done.srt');
  });

  it('strips any path so a server name cannot escape the downloads folder', () => {
    expect(dispositionName(_res('attachment; filename="../../etc/passwd"')))
      .toBe('passwd');
  });

  it('returns empty string when there is no usable header', () => {
    expect(dispositionName(_res(''))).toBe('');
    expect(dispositionName(_res('attachment'))).toBe('');
    expect(dispositionName({})).toBe('');
    expect(dispositionName(null)).toBe('');
  });
});

describe('saveTextAs', () => {
  let clicked;
  let origCreate;
  let origRevoke;

  beforeEach(() => {
    clicked = [];
    origCreate = URL.createObjectURL;
    origRevoke = URL.revokeObjectURL;
    URL.createObjectURL = vi.fn(() => 'blob:test-url');
    URL.revokeObjectURL = vi.fn();
    // Record the anchor's state AT CLICK TIME — that, not its state afterwards,
    // is what decides the saved file's name.
    HTMLAnchorElement.prototype.click = function () {
      clicked.push({
        download: this.download,
        href: this.getAttribute('href'),
        inDocument: document.body.contains(this),
      });
    };
    vi.useFakeTimers();
  });

  afterEach(() => {
    URL.createObjectURL = origCreate;
    URL.revokeObjectURL = origRevoke;
    vi.useRealTimers();
  });

  it('clicks an anchor that is IN the document and carries the filename', () => {
    saveTextAs('1\n00:00:01,000 --> 00:00:02,000\nhi\n', 'My Talk.srt');
    expect(clicked).toHaveLength(1);
    // The regression: a detached anchor, or one with an empty download
    // attribute, saved the subtitle file with no name at all.
    expect(clicked[0].inDocument).toBe(true);
    expect(clicked[0].download).toBe('My Talk.srt');
    expect(clicked[0].href).toBe('blob:test-url');
  });

  it('never leaves the download attribute empty', () => {
    saveTextAs('x', '');
    expect(clicked[0].download).toBe('download');
  });

  it('removes the anchor but does not revoke the blob in the same tick', () => {
    saveTextAs('x', 'a.srt');
    expect(document.querySelectorAll('a[download]')).toHaveLength(0);
    // Revoking synchronously can pull the blob out from under a download the
    // browser has only scheduled.
    expect(URL.revokeObjectURL).not.toHaveBeenCalled();
    vi.advanceTimersByTime(10000);
    expect(URL.revokeObjectURL).toHaveBeenCalledWith('blob:test-url');
  });
});
