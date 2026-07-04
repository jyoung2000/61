// @vitest-environment jsdom
/**
 * WAI-ARIA pattern tests (polish task 5): combobox semantics on the
 * command palette, menu semantics + Home/End on the context menu, and
 * modal-dialog semantics (focus trap + restore) on the cheat sheet.
 */
import { describe, it, expect, beforeAll, beforeEach, afterEach } from 'vitest';
import React from 'react';
import { createRoot } from 'react-dom/client';
import { act } from 'react';
import CommandPalette from './CommandPalette';
import ContextMenu from './ContextMenu';
import ShortcutCheatSheet from './ShortcutCheatSheet';

beforeAll(() => { globalThis.IS_REACT_ACT_ENVIRONMENT = true; });

let container;
let root;

beforeEach(() => {
  container = document.createElement('div');
  document.body.appendChild(container);
});

afterEach(async () => {
  await act(async () => { root?.unmount(); });
  container.remove();
  document.body.querySelectorAll('.ve-palette-backdrop, .ve-context-menu')
    .forEach((el) => el.remove());
});

const render = async (el) => {
  await act(async () => {
    root = createRoot(container);
    root.render(el);
  });
};

const key = (target, k, opts = {}) => act(async () => {
  target.dispatchEvent(new KeyboardEvent('keydown', { key: k, bubbles: true, ...opts }));
});

describe('CommandPalette combobox pattern', () => {
  it('exposes combobox → listbox → option wiring', async () => {
    await render(<CommandPalette open onClose={() => {}} ctx={{}} />);
    const input = document.body.querySelector('[role="combobox"]');
    expect(input).toBeTruthy();
    expect(input.getAttribute('aria-expanded')).toBe('true');
    const listboxId = input.getAttribute('aria-controls');
    const listbox = document.getElementById(listboxId);
    expect(listbox?.getAttribute('role')).toBe('listbox');
    const options = listbox.querySelectorAll('[role="option"]');
    expect(options.length).toBeGreaterThan(10);
    // Active row tracked via aria-activedescendant, not focus stealing
    const activeId = input.getAttribute('aria-activedescendant');
    expect(document.getElementById(activeId)?.getAttribute('aria-selected')).toBe('true');
    expect(document.activeElement === input || document.activeElement === document.body).toBe(true);
  });

  it('announces the result count politely', async () => {
    await render(<CommandPalette open onClose={() => {}} ctx={{}} />);
    const live = document.body.querySelector('[aria-live="polite"]');
    expect(live?.textContent).toMatch(/\d+ commands? available/);
  });
});

describe('ContextMenu menu pattern', () => {
  const items = [
    { id: 'a', label: 'Alpha', onSelect: () => {} },
    { id: 'b', label: 'Beta', onSelect: () => {} },
    { separator: true },
    { id: 'c', label: 'Gamma', onSelect: () => {} },
  ];

  it('menu/menuitem roles with arrow + Home/End navigation', async () => {
    await render(<ContextMenu x={10} y={10} items={items} onClose={() => {}} />);
    const menu = document.body.querySelector('[role="menu"]');
    expect(menu).toBeTruthy();
    const rows = menu.querySelectorAll('[role="menuitem"]');
    expect(rows.length).toBe(3);
    expect(menu.querySelector('[role="separator"]')).toBeTruthy();

    await key(menu, 'ArrowDown');
    expect(menu.querySelector('.ve-context-menu__item--focused')?.textContent).toContain('Alpha');
    await key(menu, 'End');
    expect(menu.querySelector('.ve-context-menu__item--focused')?.textContent).toContain('Gamma');
    await key(menu, 'Home');
    expect(menu.querySelector('.ve-context-menu__item--focused')?.textContent).toContain('Alpha');
  });
});

describe('ShortcutCheatSheet modal pattern', () => {
  it('role=dialog + aria-modal, takes focus, restores it on close', async () => {
    const opener = document.createElement('button');
    document.body.appendChild(opener);
    opener.focus();

    function Host() {
      const [open, setOpen] = React.useState(true);
      return <ShortcutCheatSheet open={open} onClose={() => setOpen(false)} />;
    }
    await render(<Host />);
    const dialog = document.body.querySelector('[role="dialog"][aria-modal="true"]');
    expect(dialog).toBeTruthy();
    expect(document.activeElement).toBe(dialog);

    // Close via the close button — focus returns to the opener
    const close = dialog.querySelector('button[aria-label="Close shortcuts"]');
    await act(async () => { close.click(); });
    expect(document.body.querySelector('[role="dialog"][aria-modal="true"]')).toBeNull();
    expect(document.activeElement).toBe(opener);
    opener.remove();
  });
});
