import { test, expect } from "@playwright/test";
import { readFileSync } from "node:fs";
import path from "node:path";

test.describe("Phases 12–17 with real API and PostgreSQL", () => {
  let first: { id: string; image_url: string };
  let second: { id: string; image_url: string };
  let jobId: string;
  test.beforeAll(async ({ request }) => {
    test.skip(
      process.env.JIGSAW_LIVE_TEST !== "1",
      "Requires the real PostgreSQL-backed API",
    );
    async function create(scorer: string) {
      const body = new FormData();
      for (let i = 0; i < 25; i++) {
        const name = `piece_${String(i).padStart(3, "0")}.png`;
        body.append(
          "pieces",
          new Blob([readFileSync(path.join("public/demo", name))], {
            type: "image/png",
          }),
          name,
        );
      }
      body.append("scorer", scorer);
      body.append("search", "greedy");
      const response = await fetch("http://127.0.0.1:3000/api/jobs", {
        method: "POST",
        body,
      });
      expect(response.status).toBe(202);
      const job = await response.json();
      jobId = job.id;
      let state;
      await expect
        .poll(async () => {
          state = await (await request.get(`/api/jobs/${job.id}`)).json();
          return state.status;
        })
        .toBe("completed");
      const completed = await (await request.get(`/api/jobs/${job.id}`)).json();
      return (
        await request.get(`/api/reconstructions/${completed.result_id}`)
      ).json();
    }
    first = await create("cnn");
    second = await create("classical");
  });

  test("keyboard, mouse drag, touch and evidence preserve all pieces", async ({
    page,
  }) => {
    await page.goto(`/?result=${first.id}`);
    const tiles = page.locator(".puzzle-canvas button");
    await expect(tiles).toHaveCount(25);
    const original = await tiles.evaluateAll((nodes) =>
      nodes.map((node) => (node as HTMLElement).dataset.piece),
    );
    await tiles.nth(0).focus();
    await page.keyboard.press("Enter");
    await tiles.nth(1).focus();
    await page.keyboard.press("Enter");
    await expect(tiles.nth(0)).toHaveAttribute("data-piece", original[1]!);
    await page.getByRole("button", { name: "Undo move" }).click();
    await expect(tiles.nth(0)).toHaveAttribute("data-piece", original[0]!);
    await tiles.nth(0).scrollIntoViewIfNeeded();
    const a = await tiles.nth(0).boundingBox(),
      b = await tiles.nth(6).boundingBox();
    await page.mouse.move(a!.x + a!.width / 2, a!.y + a!.height / 2);
    await page.mouse.down();
    await page.mouse.move(b!.x + b!.width / 2, b!.y + b!.height / 2, {
      steps: 8,
    });
    await page.mouse.up();
    await expect(tiles.nth(0)).toHaveAttribute("data-piece", original[6]!);
    expect(
      new Set(
        await tiles.evaluateAll((nodes) =>
          nodes.map((node) => (node as HTMLElement).dataset.piece),
        ),
      ).size,
    ).toBe(25);
    await page.getByRole("button", { name: "Restore solver result" }).click();
    await page
      .getByRole("checkbox", {
        name: "Show relative edge evidence",
        exact: false,
      })
      .check();
    await expect(page.locator(".evidence-tint")).toHaveCount(25);
    await tiles.nth(0).click();
    await tiles.nth(1).click();
    await expect(page.locator(".evidence-tint")).toHaveCount(0);
    await expect(
      page.getByRole("region", { name: "Compatibility evidence" }),
    ).toContainText("not calibrated");
    const context = await page
      .context()
      .browser()!
      .newContext({
        viewport: { width: 390, height: 844 },
        hasTouch: true,
        isMobile: true,
      });
    try {
      const mobile = await context.newPage();
      await mobile.goto(`http://127.0.0.1:3000/?result=${first.id}`);
      const touchTiles = mobile.locator(".puzzle-canvas button");
      await touchTiles.nth(0).tap();
      await touchTiles.nth(1).tap();
      await expect(touchTiles.nth(0)).toHaveAttribute(
        "data-piece",
        original[1]!,
      );
      expect(
        await mobile.evaluate(
          () => document.documentElement.scrollWidth <= innerWidth,
        ),
      ).toBe(true);
    } finally {
      await context.close();
    }
  });

  test("save a correction, reload it, compare and download ZIP", async ({
    page,
  }) => {
    await page.goto(`/?result=${first.id}`);
    const tiles = page.locator(".puzzle-canvas button");
    await expect(tiles).toHaveCount(25);
    const before = await tiles.nth(1).getAttribute("data-piece");
    await tiles.nth(0).click();
    await tiles.nth(1).click();
    await page.getByLabel("Optional note").fill("Browser-tested correction");
    const savedResponse = page.waitForResponse(
      (response) =>
        response.url().endsWith("/corrections") &&
        response.request().method() === "POST",
    );
    await page
      .getByRole("button", { name: "Save correction", exact: true })
      .click();
    const saved = await (await savedResponse).json();
    await expect(page.getByLabel("Load saved correction")).toHaveValue(
      saved.id,
    );
    await expect(
      page.getByRole("img", { name: "Saved human correction comparison" }),
    ).toBeVisible();
    await page.reload();
    await expect(
      page
        .getByLabel("Load saved correction")
        .locator(`option[value="${saved.id}"]`),
    ).toHaveCount(1);
    await page.getByLabel("Load saved correction").selectOption(saved.id);
    await expect(tiles.nth(0)).toHaveAttribute("data-piece", before!);
    const downloaded = page.waitForEvent("download");
    await page.getByRole("link", { name: "Download ZIP correction" }).click();
    expect((await downloaded).suggestedFilename()).toBe(`${saved.id}.zip`);
    await page.screenshot({
      path: "../artifacts/frontend/phases12-17.png",
      fullPage: true,
    });
  });

  test("history comparison, exports and a resumable job URL", async ({
    page,
  }) => {
    await page.goto(`/?job=${jobId}`);
    await expect(page).toHaveURL(`http://127.0.0.1:3000/?result=${second.id}`);
    await page.getByLabel(`Compare attempt ${first.id}`).check();
    await page.getByLabel(`Compare attempt ${second.id}`).check();
    await expect(
      page.getByRole("heading", { name: "Side-by-side comparison" }),
    ).toBeVisible();
    await expect(
      page.getByRole("region", { name: "Project analytics" }),
    ).toHaveCount(0);
    const report = await (
      await page.request.get(
        `/api/reconstructions/${first.id}/export?format=json`,
      )
    ).json();
    expect(report.ground_truth_available).toBe(false);
    expect(report.position_accuracy).toBeNull();
    expect(report.result.diagnostics.edges).toHaveLength(40);
  });

  test("reconstruct again sends the edited grid and creates a new attempt", async ({
    page,
  }) => {
    await page.goto(`/?result=${first.id}`);
    const tiles = page.locator(".puzzle-canvas button");
    await expect(tiles).toHaveCount(25);
    await tiles.nth(0).click();
    await tiles.nth(1).click();
    const edited = await tiles.evaluateAll((nodes) =>
      nodes.map((node) => Number((node as HTMLElement).dataset.piece)),
    );
    await page.getByText("Solver options", { exact: true }).click();
    await page
      .getByLabel("Reconstruction method", { exact: true })
      .selectOption("classical");
    await page
      .getByLabel("Reconstruction search", { exact: true })
      .selectOption("greedy");
    const submitted = page.waitForResponse(
      (response) =>
        response.url().endsWith(`/reconstructions/${first.id}/reconstruct`) &&
        response.request().method() === "POST",
    );
    await page
      .getByRole("region", { name: "Interactive puzzle", exact: true })
      .getByRole("button", { name: "Reconstruct again", exact: true })
      .click();
    const response = await submitted;
    expect(response.status()).toBe(202);
    expect(response.request().postDataJSON().order).toEqual(edited);
    await expect(page).toHaveURL(
      (url) =>
        Boolean(url.searchParams.get("result")) &&
        url.searchParams.get("result") !== first.id,
    );
    const id = new URL(page.url()).searchParams.get("result");
    expect(id).not.toBe(first.id);
    const result = await (
      await page.request.get(`/api/reconstructions/${id}`)
    ).json();
    expect(result.source_result_id).toBe(first.id);
    expect(result.source_order).toEqual(edited);
    expect(result.scorer).toBe("classical");
    expect(
      new Set(
        result.arrangement.map(
          (piece: { upload_index: number }) => piece.upload_index,
        ),
      ).size,
    ).toBe(25);
    expect(
      (await page.request.get(`/api/reconstructions/${first.id}`)).status(),
    ).toBe(200);
    await expect(
      page.getByRole("img", { name: "Reconstructed 5 by 5 puzzle" }),
    ).toBeVisible();
  });

  test("a rejected rerun leaves edits in place and enables retry", async ({
    page,
  }) => {
    await page.goto(`/?result=${first.id}`);
    const tiles = page.locator(".puzzle-canvas button");
    await expect(tiles).toHaveCount(25);
    await tiles.nth(0).click();
    await tiles.nth(1).click();
    const edited = await tiles.evaluateAll((nodes) =>
      nodes.map((node) => (node as HTMLElement).dataset.piece),
    );
    await page.route(`**/reconstructions/${first.id}/reconstruct`, (route) =>
      route.fulfill({
        status: 429,
        json: { detail: "The local queue is full; retry shortly" },
      }),
    );
    await page
      .getByRole("button", { name: "Reconstruct again", exact: true })
      .click();
    await expect(
      page
        .getByRole("region", { name: "Reconstruct current pieces" })
        .getByRole("alert"),
    ).toContainText("queue is full");
    await expect(
      page.getByRole("button", { name: "Reconstruct again", exact: true }),
    ).toBeEnabled();
    expect(
      await tiles.evaluateAll((nodes) =>
        nodes.map((node) => (node as HTMLElement).dataset.piece),
      ),
    ).toEqual(edited);
  });
});
