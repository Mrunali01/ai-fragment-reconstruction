import { test, expect } from "@playwright/test";
import path from "node:path";
import { readFileSync, writeFileSync } from "node:fs";

const demoFile = path.join(process.cwd(), "public/demo/piece_000.png");
const demoFiles = Array.from({ length: 25 }, (_, index) =>
  path.join(
    process.cwd(),
    `public/demo/piece_${String(index).padStart(3, "0")}.png`,
  ),
);

test("validates fragment count and offers a keyboard accessible upload", async ({
  page,
}) => {
  await page.route("**/api/health", (route) =>
    route.fulfill({
      json: { status: "ok", cnn_available: true, classical_available: true },
    }),
  );
  await page.goto("/");
  await expect(page.getByRole("heading", { level: 1 })).toBeVisible();
  await expect(
    page.getByRole("button", { name: "Reconstruct puzzle" }),
  ).toBeDisabled();
  await page.getByLabel("Choose 25 fragments").setInputFiles([demoFile]);
  await expect(page.getByRole("main").getByRole("alert")).toContainText(
    "Choose exactly 25 fragments",
  );
  await page.getByRole("button", { name: "Drop your fragments here" }).focus();
  await expect(
    page.getByRole("button", { name: "Drop your fragments here" }),
  ).toBeFocused();
  const wrongSize = await page.evaluate(() => {
    const canvas = document.createElement("canvas");
    canvas.width = 32;
    canvas.height = 32;
    return Array.from(atob(canvas.toDataURL("image/png").split(",")[1]), (c) =>
      c.charCodeAt(0),
    );
  });
  await page.getByLabel("Choose 25 fragments").setInputFiles([
    {
      name: "wrong.png",
      mimeType: "image/png",
      buffer: Buffer.from(wrongSize),
    },
    ...demoFiles.slice(1).map((name) => ({
      name: path.basename(name),
      mimeType: "image/png",
      buffer: readFileSync(name),
    })),
  ]);
  await expect(page.getByRole("main").getByRole("alert")).toContainText(
    "64 × 64 pixels",
  );
});

test("mobile sample previews, loading, server failure and retry", async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.route("**/api/health", (route) =>
    route.fulfill({
      json: { status: "ok", cnn_available: true, classical_available: true },
    }),
  );
  let requests = 0;
  await page.route("**/api/reconstructions", async (route) => {
    requests++;
    await new Promise((resolve) => setTimeout(resolve, 250));
    await route.fulfill({
      status: 503,
      json: { detail: "Solver is busy; retry shortly" },
    });
  });
  await page.goto("/");
  await page.getByRole("button", { name: "Try a sample puzzle" }).click();
  await expect(page.getByRole("img", { name: /^Fragment / })).toHaveCount(25);
  await page.getByRole("button", { name: "Reconstruct puzzle" }).click();
  await expect(
    page.getByRole("button", { name: "Assembling your picture" }),
  ).toBeDisabled();
  await expect(page.getByRole("main").getByRole("alert")).toHaveText(
    "Solver is busy; retry shortly",
  );
  await expect(
    page.getByRole("button", { name: "Reconstruct puzzle" }),
  ).toBeEnabled();
  await page.getByRole("button", { name: "Reconstruct puzzle" }).click();
  await expect.poll(() => requests).toBe(2);
  await expect(page.getByRole("main").getByRole("alert")).toBeVisible();
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= window.innerWidth,
    ),
  ).toBe(true);
  await page.screenshot({
    path: "../artifacts/frontend/mobile.png",
    fullPage: true,
  });
});

test("offline and unavailable CNN states are actionable", async ({ page }) => {
  await page.route("**/api/health", (route) =>
    route.fulfill({ status: 503, body: "unavailable" }),
  );
  await page.goto("/");
  await expect(page.getByText("Solver offline")).toBeVisible();
  await page.unroute("**/api/health");
  await page.route("**/api/health", (route) =>
    route.fulfill({
      json: { status: "ok", cnn_available: false, classical_available: true },
    }),
  );
  await page.getByRole("button", { name: "Refresh solver status" }).click();
  await expect(
    page.getByText("The trained CNN is unavailable.", { exact: false }),
  ).toBeVisible();
  await page.getByLabel("Matching method").selectOption("classical");
  await page.getByLabel("Choose 25 fragments").setInputFiles(demoFiles);
  await expect(
    page.getByRole("button", { name: "Reconstruct puzzle" }),
  ).toBeEnabled();
});

test("real CNN reconstruction persists across a page reload", async ({
  page,
}) => {
  test.skip(
    process.env.JIGSAW_LIVE_TEST !== "1",
    "Start the real PostgreSQL-backed API and set JIGSAW_LIVE_TEST=1",
  );
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto("/");
  await expect(page.getByText("Solver connected")).toBeVisible();
  await page.getByRole("button", { name: "Try a sample puzzle" }).click();
  await expect(page.getByRole("img", { name: /^Fragment / })).toHaveCount(25);
  const responsePromise = page.waitForResponse(
    (response) =>
      response.url().endsWith("/api/reconstructions") &&
      response.request().method() === "POST",
  );
  await page.getByRole("button", { name: "Reconstruct puzzle" }).click();
  const response = await responsePromise;
  expect(response.status()).toBe(201);
  const result = await response.json();
  expect(
    new Set(
      result.arrangement.map(
        (piece: { upload_index: number }) => piece.upload_index,
      ),
    ).size,
  ).toBe(25);
  await expect(
    page.getByRole("img", { name: "Reconstructed 5 by 5 puzzle" }),
  ).toBeVisible();
  await expect
    .poll(() =>
      page
        .getByRole("img", { name: "Reconstructed 5 by 5 puzzle" })
        .evaluate((image: HTMLImageElement) => image.naturalWidth),
    )
    .toBe(320);
  await expect(page).toHaveURL(/\?result=/);
  await page.reload();
  await expect(
    page.getByRole("img", { name: "Reconstructed 5 by 5 puzzle" }),
  ).toBeVisible();
  await expect(page.getByRole("link", { name: "Save image" })).toHaveAttribute(
    "href",
    `/api${result.image_url}`,
  );
  expect(errors).toEqual([]);
  await page.screenshot({
    path: "../artifacts/frontend/desktop-result.png",
    fullPage: true,
  });
  writeFileSync(
    "../artifacts/frontend/live-result.json",
    JSON.stringify(result, null, 2),
  );
});
