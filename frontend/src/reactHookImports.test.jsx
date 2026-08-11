// @vitest-environment node
/**
 * Every React hook a file USES must be IMPORTED in that file.
 *
 * There is no ESLint in this project, so a missing hook import is caught by
 * nothing until the component renders and throws
 * "ReferenceError: useCallback is not defined" — a white page for whoever
 * opens that route. (Real near-miss: `useCallback` was added to Settings.jsx
 * while the import line still read `{ useState, useEffect, useRef, useMemo }`;
 * the whole test suite passed because no test renders that page.)
 *
 * Scans every source file for calls to React's built-in hooks and asserts the
 * hook is named in an import from 'react'.
 */
import { describe, it, expect } from 'vitest';
import { readFileSync, readdirSync, statSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const SRC = path.dirname(fileURLToPath(import.meta.url));

const REACT_HOOKS = [
  'useState', 'useEffect', 'useLayoutEffect', 'useRef', 'useMemo',
  'useCallback', 'useContext', 'useReducer', 'useId', 'useTransition',
  'useDeferredValue', 'useSyncExternalStore', 'useImperativeHandle',
  'useDebugValue', 'useInsertionEffect', 'useOptimistic', 'useActionState',
];

function sourceFiles(dir, out = []) {
  for (const name of readdirSync(dir)) {
    if (name === 'node_modules' || name.startsWith('.')) continue;
    const full = path.join(dir, name);
    if (statSync(full).isDirectory()) sourceFiles(full, out);
    else if (/\.jsx?$/.test(name) && !/\.test\.jsx?$/.test(name)) out.push(full);
  }
  return out;
}

/** Hook names imported from 'react' (named imports, plus `React.useX` usage). */
function importedHooks(src) {
  const named = new Set();
  for (const m of src.matchAll(/import\s+(?:React\s*,\s*)?\{([^}]*)\}\s*from\s*['"]react['"]/g)) {
    for (const part of m[1].split(',')) {
      const name = part.trim().split(/\s+as\s+/)[0].trim();
      if (name) named.add(name);
    }
  }
  return named;
}

describe('React hook imports', () => {
  const files = sourceFiles(SRC);

  it('scans a meaningful number of source files', () => {
    expect(files.length).toBeGreaterThan(20);
  });

  it('every React hook used is imported from react', () => {
    const problems = [];
    for (const file of files) {
      const src = readFileSync(file, 'utf8');
      const imported = importedHooks(src);
      for (const hook of REACT_HOOKS) {
        // A bare `useX(` call — not `React.useX(`, not `.useX(`, and not a
        // definition of a same-named local/custom hook.
        const used = new RegExp(`(^|[^.\\w])${hook}\\s*\\(`, 'm').test(src);
        if (!used) continue;
        const defined = new RegExp(`(function|const|let)\\s+${hook}\\b`).test(src);
        if (defined || imported.has(hook)) continue;
        problems.push(`${path.relative(SRC, file)} uses ${hook} without importing it`);
      }
    }
    expect(problems, problems.join('\n')).toEqual([]);
  });
});
