"""Smoke-test the browser camera controls and incident log with a fake camera."""

from __future__ import annotations

import json
from pathlib import Path

from playwright.sync_api import sync_playwright


def main() -> None:
    chrome = Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe")
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            headless=True,
            executable_path=str(chrome) if chrome.exists() else None,
            args=[
                "--use-fake-device-for-media-stream",
                "--use-fake-ui-for-media-stream",
            ],
        )
        page = browser.new_page(
            viewport={"width": 1440, "height": 900},
            permissions=["camera"],
        )
        page.goto("http://127.0.0.1:8000", wait_until="domcontentloaded")
        page.wait_for_function("yoloReady && poseReady", timeout=180_000)
        page.locator("#cameraSourceType").select_option("computer")
        page.locator("#startBtn").click()
        page.wait_for_function(
            "sessionActive && sourceMode === 'camera'", timeout=30_000
        )
        page.wait_for_function(
            "phoneState && phoneState.textContent.includes('·')", timeout=30_000
        )

        result = {
            "camera": page.evaluate(
                """() => ({
                    status: cameraStatus.textContent,
                    info: cameraInfo.textContent,
                    resolution: `${video.videoWidth}x${video.videoHeight}`
                })"""
            )
        }
        result["phone_pipeline"] = page.evaluate(
            """() => ({
                state: phoneState.textContent,
                intervalMs: PHONE_INFERENCE_INTERVAL_MS,
                firstTile: getNextPhoneTile(),
                transformedBox: transformPhoneBoxes(
                    [{x1: 100, y1: 100, x2: 200, y2: 200, confidence: 0.9}],
                    {xStart: 0.44, yStart: 0.42, xScale: 0.56, yScale: 0.58}
                )[0]
            })"""
        )
        page.evaluate(
            """() => registerViolation(
                "TEST", "Cảnh báo thử", "Kiểm tra nhật ký ảnh minh chứng"
            )"""
        )
        result["log"] = page.evaluate(
            """() => ({
                count: incidentTotal,
                cards: document.querySelectorAll('.incident-card').length,
                visible: logFeed.offsetParent !== null
            })"""
        )
        page.locator("#stopBtn").click()
        result["stopped"] = page.evaluate(
            "() => ({ active: sessionActive, status: cameraStatus.textContent })"
        )
        page.locator('[data-workspace-mode="video"]').click()
        result["video_screen"] = page.evaluate(
            """() => ({
                title: workspaceTitle.textContent,
                liveHidden: liveSourcePanel.classList.contains('hidden'),
                videoVisible: !videoSourcePanel.classList.contains('hidden'),
                stopCameraHidden: stopBtn.classList.contains('hidden'),
                stopVideoHidden: stopVideoBtn.classList.contains('hidden')
            })"""
        )
        browser.close()

    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
