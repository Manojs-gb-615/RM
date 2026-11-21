import { test, expect } from '@playwright/test';
import path from 'path';
import { fileURLToPath } from 'url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const FIXTURES = path.join(HERE, 'fixtures');
const SHOTS = 'tests/e2e/screenshots';

test('studio layout, retrieval, and frame-level video sync', async ({ page }) => {
  await page.goto('/');
  await expect(page.getByText('SignLingo AI Mini')).toBeVisible();
  await page.screenshot({ path: `${SHOTS}/01-initial.png`, fullPage: true });

  // Layout: LEFT video, RIGHT skeleton, BOTTOM transcript + controls.
  const videoHeading = await page.getByRole('heading', { name: /Educational Video/ }).boundingBox();
  const poseHeading = await page.getByRole('heading', { name: /ASL Pose Viewer/ }).boundingBox();
  const transcriptHeading = await page.getByRole('heading', { name: /Transcript & Controls/ }).boundingBox();
  expect(videoHeading).not.toBeNull();
  expect(poseHeading).not.toBeNull();
  expect(transcriptHeading).not.toBeNull();
  expect(videoHeading!.x).toBeLessThan(poseHeading!.x);
  expect(transcriptHeading!.y).toBeGreaterThan(videoHeading!.y + 100);

  await page.locator('[data-testid=video-input]').setInputFiles(path.join(FIXTURES, 'sample.mp4'));
  await page.locator('[data-testid=transcript-input]').setInputFiles(path.join(FIXTURES, 'sample.srt'));
  await expect(page.getByTestId('sentence-0')).toBeVisible();
  await expect(page.getByTestId('video')).toBeVisible();
  await page.screenshot({ path: `${SHOTS}/02-transcript-loaded.png`, fullPage: true });

  await page.getByTestId('generate-btn').click();
  await expect(page.getByTestId('pose-canvas')).toHaveAttribute('data-total', /^[1-9]\d*$/);
  await page.screenshot({ path: `${SHOTS}/03-after-search.png`, fullPage: true });

  const video = page.getByTestId('video');
  await video.evaluate(async (el) => {
    const v = el as HTMLVideoElement;
    v.muted = true;
    v.currentTime = 0;
    await v.play();
  });
  await page.waitForTimeout(800);
  await video.evaluate((el) => ((el as HTMLVideoElement).pause()));

  // Pause must freeze the animation at the same frame.
  const pausedState = await page.evaluate(() => {
    const v = document.querySelector('[data-testid=video]') as HTMLVideoElement;
    const c = document.querySelector('[data-testid=pose-canvas]') as HTMLCanvasElement;
    return { paused: v.paused, frame: Number(c.dataset.frame ?? -1) };
  });
  expect(pausedState.paused).toBe(true);
  await page.waitForTimeout(500);
  const pausedFrameLater = await page.evaluate(
    () => Number((document.querySelector('[data-testid=pose-canvas]') as HTMLCanvasElement).dataset.frame ?? -1),
  );
  expect(pausedFrameLater).toBe(pausedState.frame);

  await expect
    .poll(async () => page.evaluate(() => {
      const v = document.querySelector('[data-testid=video]') as HTMLVideoElement;
      const c = document.querySelector('[data-testid=pose-canvas]') as HTMLCanvasElement;
      const a = document.querySelector('[data-testid=active-sentence]') as HTMLElement | null;
      return { t: v.currentTime, total: Number(c?.dataset.total || 0), frame: Number(c?.dataset.frame ?? -1), hasSeg: !!a };
    }), { timeout: 5000 })
    .toMatchObject({ hasSeg: true });

  const sync = await page.evaluate(() => {
    const v = document.querySelector('[data-testid=video]') as HTMLVideoElement;
    const c = document.querySelector('[data-testid=pose-canvas]') as HTMLCanvasElement;
    const a = document.querySelector('[data-testid=active-sentence]') as HTMLElement | null;
    const start = parseFloat(a?.dataset.start || '0');
    const end = parseFloat(a?.dataset.end || '1');
    const total = Number(c?.dataset.total || 0);
    const frame = Number(c?.dataset.frame ?? -1);
    const progress = Math.min(1, Math.max(0, (v.currentTime - start) / Math.max(0.001, end - start)));
    const expected = Math.min(total - 1, Math.floor(progress * total));
    return { t: v.currentTime, start, end, total, frame, expected };
  });
  expect(sync.total).toBeGreaterThan(0);
  expect(Math.abs(sync.frame - sync.expected), JSON.stringify(sync)).toBeLessThanOrEqual(1);
  await page.screenshot({ path: `${SHOTS}/04-during-playback.png`, fullPage: true });

  // Seek: jump to the middle of the active segment and check the frame again.
  const seek = await page.evaluate(async () => {
    const v = document.querySelector('[data-testid=video]') as HTMLVideoElement;
    const a = document.querySelector('[data-testid=active-sentence]') as HTMLElement;
    const start = parseFloat(a.dataset.start || '0');
    const end = parseFloat(a.dataset.end || '1');
    v.currentTime = start + 0.5 * (end - start);
    await new Promise((r) => setTimeout(r, 400));
    const c = document.querySelector('[data-testid=pose-canvas]') as HTMLCanvasElement;
    const total = Number(c.dataset.total || 0);
    const frame = Number(c.dataset.frame ?? -1);
    return { total, frame, expected: Math.min(total - 1, Math.floor(0.5 * total)) };
  });
  expect(Math.abs(seek.frame - seek.expected)).toBeLessThanOrEqual(1);

  // Playback-rate change must not break the mapping.
  await video.evaluate((el) => { (el as HTMLVideoElement).playbackRate = 2; });
  const afterRate = await page.evaluate(() => {
    const c = document.querySelector('[data-testid=pose-canvas]') as HTMLCanvasElement;
    return { total: Number(c.dataset.total || 0), frame: Number(c.dataset.frame ?? -1) };
  });
  expect(afterRate.frame).toBeGreaterThanOrEqual(0);
  expect(afterRate.frame).toBeLessThan(afterRate.total);

  // Loop must wrap back to the start without breaking the frame mapping.
  const loopInfo = await video.evaluate(async (el) => {
    const v = el as HTMLVideoElement;
    v.playbackRate = 1;
    v.currentTime = Math.max(0, v.duration - 0.25);
    await v.play();
    await new Promise((r) => setTimeout(r, 1800));
    const c = document.querySelector('[data-testid=pose-canvas]') as HTMLCanvasElement;
    return {
      loop: v.loop,
      t: v.currentTime,
      frame: Number(c.dataset.frame ?? -1),
      total: Number(c.dataset.total || 0),
    };
  });
  expect(loopInfo.loop).toBe(true);
  expect(loopInfo.t, JSON.stringify(loopInfo)).toBeLessThan(3);
  expect(loopInfo.frame).toBeGreaterThanOrEqual(0);
  expect(loopInfo.frame).toBeLessThan(loopInfo.total);
  await video.evaluate((el) => ((el as HTMLVideoElement).pause()));

  // Clicking a transcript sentence seeks the video.
  await page.getByTestId('sentence-1').click();
  await page.waitForTimeout(400);
  const seeked = await video.evaluate((el) => (el as HTMLVideoElement).currentTime);
  expect(seeked).toBeGreaterThanOrEqual(2.4);
  await page.screenshot({ path: `${SHOTS}/05-after-sentence-click.png`, fullPage: true });

  // Real (non-synthetic) index should not show the synthetic banner.
  await expect(page.getByText('SYNTHETIC DATA')).toHaveCount(0);
});

test('synthetic data banner is impossible to miss', async ({ page }) => {
  await page.route('**/api/status', (route) => route.fulfill({
    status: 200,
    contentType: 'application/json',
    body: JSON.stringify({
      ready: true,
      num_indexed: 30,
      dataset_samples: 30,
      dataset_with_pose: 30,
      is_synthetic: true,
      is_trained: false,
      message: 'Retrieval engine ready.',
    }),
  }));
  await page.goto('/');
  await expect(page.getByText('SYNTHETIC DATA')).toBeVisible();
  await page.screenshot({ path: `${SHOTS}/06-synthetic-banner.png`, fullPage: true });
});
