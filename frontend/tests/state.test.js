import test from 'node:test';
import assert from 'node:assert/strict';
import { state } from '../src/state.js';

test('browser session starts without retained user data', () => {
  assert.equal(state.session, null);
  assert.equal(state.stopped, false);
});

test('API and components do not use browser persistent storage or HTML injection', async () => {
  const { readFile } = await import('node:fs/promises');
  for (const name of ['api.js', 'app.js', 'components/chat.js', 'components/approval.js', 'components/results.js']) {
    const source = await readFile(new URL(`../src/${name}`, import.meta.url), 'utf8');
    assert.doesNotMatch(source, /localStorage\.|sessionStorage\.|indexedDB\.|\.innerHTML\s*=/);
  }
});

test('Vite exposes only the explicit public configuration allowlist', async () => {
  const { default: vite } = await import('../vite.config.js');
  assert.equal(vite.envDir, false);
  assert.deepEqual(vite.envPrefix, []);
  const settings = JSON.parse(vite.define.__APP_CONFIG__);
  assert.deepEqual(Object.keys(settings).sort(), ['apiBaseUrl', 'appTitle', 'maxUploadMb']);
});
