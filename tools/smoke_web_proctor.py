"""Browser smoke test for the standalone web proctor.

Usage:
    python tools/smoke_web_proctor.py VIDEO [--url http://127.0.0.1:8001]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from playwright.sync_api import sync_playwright


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("--url", default="http://127.0.0.1:8001")
    parser.add_argument("--seek", type=float, default=20.0)
    parser.add_argument("--wait-ms", type=int, default=5_000)
    parser.add_argument("--trace-phones", action="store_true")
    parser.add_argument("--screenshot", type=Path)
    parser.add_argument("--browser", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video_path = args.video.resolve()
    if not video_path.exists():
        raise SystemExit(f"Video not found: {video_path}")

    console_errors: list[str] = []
    page_errors: list[str] = []
    with sync_playwright() as playwright:
        installed_browser = args.browser
        if installed_browser is None:
            candidates = [
                Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
                Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
            ]
            installed_browser = next((path for path in candidates if path.exists()), None)
        launch_options = {"headless": True}
        if installed_browser:
            launch_options["executable_path"] = str(installed_browser)
        browser = playwright.chromium.launch(**launch_options)
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        page.on("console", lambda message: console_errors.append(message.text)
                if message.type == "error" else None)
        page.on("pageerror", lambda error: page_errors.append(str(error)))
        page.goto(args.url, wait_until="domcontentloaded", timeout=30_000)
        try:
            page.wait_for_function("yoloReady && poseReady", timeout=180_000)
        except Exception:
            print("PAGE_ERRORS")
            for error in page_errors:
                print(error)
            print("CONSOLE_ERRORS")
            for error in console_errors:
                print(error)
            raise
        page.locator('[data-workspace-mode="video"]').click()
        page.locator("#videoFileInput").set_input_files(str(video_path))
        page.locator("#analyzeVideoBtn").click()
        try:
            page.wait_for_function("sessionActive && sourceMode === 'video'", timeout=30_000)
        except Exception:
            print("VIDEO_START_STATE")
            print(json.dumps(page.evaluate("""() => ({
                selected: selectedVideoName.textContent,
                disabled: analyzeVideoBtn.disabled,
                sourceMode,
                sessionActive,
                phoneState: phoneState.textContent
            })"""), ensure_ascii=True))
            print("PAGE_ERRORS")
            for error in page_errors:
                print(error)
            print("CONSOLE_ERRORS")
            for error in console_errors:
                print(error)
            raise
        page.evaluate(
            """async (second) => {
                video.currentTime = Math.min(second, Math.max(0, video.duration - 1));
                await video.play();
            }""",
            args.seek,
        )
        phone_trace: list[dict] = []
        if args.trace_phones:
            elapsed = 0
            while elapsed < args.wait_ms:
                page.wait_for_timeout(min(250, args.wait_ms - elapsed))
                elapsed += 250
                boxes = page.evaluate(
                    """() => getPhoneBoxes().map((box) => ({
                        time: video.currentTime,
                        confidence: box.confidence,
                        detector: box.detector || "general",
                        x1: Math.round(box.x1), y1: Math.round(box.y1),
                        x2: Math.round(box.x2), y2: Math.round(box.y2)
                    }))"""
                )
                phone_trace.extend(boxes)
        else:
            page.wait_for_timeout(args.wait_ms)

        state = page.evaluate(
            """() => ({
                people: currentBoundingBoxes.filter((box) => box.classId === 0).length,
                phones: getPhoneBoxes().length,
                phoneBoxes: getPhoneBoxes().map((box) => ({
                    confidence: box.confidence,
                    detector: box.detector || "general",
                    x1: Math.round(box.x1),
                    y1: Math.round(box.y1),
                    x2: Math.round(box.x2),
                    y2: Math.round(box.y2),
                    owner: box.associatedTrackId || null
                })),
                documents: currentBoundingBoxes.filter((box) => box.classId === 73).length,
                poses: currentPosePeople.length,
                poseHeads: currentPosePeople.filter((person) => person.headDetected).length,
                currentIds: currentBoundingBoxes
                    .filter((box) => box.classId === 0)
                    .map((box) => box.trackId)
                    .filter(Boolean)
                    .sort((a, b) => a - b),
                nextPersonTrackId,
                tracks: [...personTracks.values()].map((track) => ({
                    id: track.id,
                    direction: track.awayDirection,
                    awayMs: track.awaySince ? performance.now() - track.awaySince : 0,
                    phone: track.phoneAlerted
                })),
                incidents: incidentTotal,
                violationTypes: Object.keys(lastViolationTimes),
                phoneReady,
                phoneBusy,
                yoloState: yoloState ? yoloState.textContent : null,
                poseState: faceState ? faceState.textContent : null
            })"""
        )

        if args.screenshot:
            args.screenshot.parent.mkdir(parents=True, exist_ok=True)
            page.screenshot(path=str(args.screenshot), full_page=True)
        browser.close()

    if args.trace_phones:
        state["phoneTrace"] = phone_trace

    print(json.dumps(state, ensure_ascii=False, indent=2))
    if console_errors:
        print("CONSOLE_ERRORS")
        for error in console_errors:
            print(error)


if __name__ == "__main__":
    main()
