/* Headless smoke test for mixtriad_studio.html (run from the repository root):
 *   npm install jsdom --no-save && node tests/studio_smoke.js
 * Loads the shipped single-file browser tool, walks the embedded demo corpus
 * through all three stages, and asserts the pinned demo results plus the
 * built-in oracle. Exits non-zero on any deviation. */
"use strict";
const { JSDOM } = require("jsdom");
const fs = require("fs");
const path = require("path");

const file = path.join(process.cwd(), "mixtriad_studio.html");
const html = fs.readFileSync(file, "utf8");
const dom = new JSDOM(html, { runScripts: "dangerously", pretendToBeVisual: true,
  url: "file:///mixtriad_studio.html" });
const { window } = dom;
window.alert = m => { throw new Error("unexpected alert: " + m); };
window.URL.createObjectURL = () => "blob:fake";

function fail(msg) { console.error("FAIL:", msg); process.exit(1); }
function ok(msg) { console.log("PASS:", msg); }

setTimeout(() => {
  try {
    const $ = id => window.document.getElementById(id);
    $("btn-demo").click();
    if (!/388 cases/.test($("data-status").textContent)) fail("demo corpus did not load");
    ok("demo corpus loaded (388 cases)");

    $("btn-run-s1").click();
    if (!/R² = 0\.757/.test($("s1-out").textContent))
      fail("Stage 1 R² drifted from the pinned demo value 0.757");
    ok("Stage 1: OLS R² = 0.757");

    $("btn-run-s2").click();
    setTimeout(() => {
      try {
        const keep = $("s2-out").querySelector(".keep");
        if (!keep) fail("Stage 2 produced no retained set");
        const retained = keep.textContent.split("·").map(x => x.trim()).sort();
        const expected = ["account_topic_focus", "creator_reach",
                          "emotional_intensity", "video_duration"].sort();
        if (JSON.stringify(retained) !== JSON.stringify(expected))
          fail("Stage 2 retained set drifted: " + retained.join(", "));
        ok("Stage 2: retained the four pinned conditions");

        $("btn-run-s3").click();
        const s3 = $("s3-out").textContent;
        if (!/Truth table \(16 retained/.test(s3)) fail("truth table is not 16 rows");
        if (!/intermediate solution — consistency 0\.908, coverage 0\.530/.test(s3))
          fail("intermediate solution statistics drifted");
        ok("Stage 3 (Y): 16-row truth table, intermediate cons 0.908 / cov 0.530");

        $("dir-negated").click();
        if (!/Truth table \(16 retained/.test($("s3-out").textContent))
          fail("negated direction did not render");
        ok("Stage 3 (~Y): rendered");
        if (window.document.querySelectorAll("#s3-out svg").length < 1)
          fail("configuration chart SVG missing");
        ok("configuration chart present");

        $("btn-check").click();
        const chk = $("check-out").textContent;
        if (/FAIL/.test(chk) || !/255 cases, 0 failures/.test(chk))
          fail("built-in oracle failed:\n" + chk);
        ok("built-in oracle: all checks passed");
        console.log("studio smoke test PASSED");
        process.exit(0);
      } catch (e) { fail(e.message); }
    }, 500);
  } catch (e) { fail(e.message); }
}, 300);
