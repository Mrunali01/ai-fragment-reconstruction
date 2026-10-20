export type Scorer = "cnn" | "classical";
export type Search = "greedy" | "swaps" | "groups" | "beam";
export type Health = {
  status: "ok";
  cnn_available: boolean;
  classical_available: boolean;
};
export type Result = {
  id: string;
  status: "completed";
  created_at: string;
  scorer: Scorer;
  search: Search;
  image_url: string;
  source_result_id?: string | null;
  source_order?: number[] | null;
  diagnostics?: Diagnostics | null;
  arrangement: {
    row: number;
    column: number;
    upload_index: number;
    filename: string;
    sha256: string;
  }[];
};

export type EdgeEvidence = {
  piece: number;
  neighbor: number;
  direction: string;
  cost: number;
  rank: number;
  margin: number;
  alternatives: { piece: number; cost: number }[];
};
export type Diagnostics = {
  edges: EdgeEvidence[];
  top_ranked_border_fraction: number;
  mean_border_cost: number;
  mean_margin: number;
  warning: string;
  definition: string;
};

export async function apiResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let message = `Request failed (${response.status}). Please try again.`;
    try {
      const body = await response.json();
      if (typeof body.detail === "string") message = body.detail;
      else if (Array.isArray(body.detail))
        message = body.detail
          .map((item: { msg?: string }) => item.msg ?? "Invalid input")
          .join("; ");
    } catch {
      /* An unavailable proxy can return plain text instead of JSON. */
    }
    throw new Error(message);
  }
  return response.json() as Promise<T>;
}

export async function waitForJob(
  id: string,
  signal?: AbortSignal,
  onStatus?: (status: string) => void,
): Promise<Result> {
  while (true) {
    const job = await apiResponse<{
      status: string;
      result_id: string | null;
      error: string | null;
    }>(await fetch(`/api/jobs/${id}`, { cache: "no-store", signal }));
    onStatus?.(job.status);
    if (job.status === "failed")
      throw new Error(job.error ?? "Reconstruction failed. Please retry.");
    if (job.status === "completed" && job.result_id)
      return apiResponse<Result>(
        await fetch(`/api/reconstructions/${job.result_id}`, {
          cache: "no-store",
          signal,
        }),
      );
    await new Promise<void>((resolve, reject) => {
      const onAbort = () => {
        clearTimeout(timer);
        reject(new DOMException("Aborted", "AbortError"));
      };
      const timer = setTimeout(() => {
        signal?.removeEventListener("abort", onAbort);
        resolve();
      }, 500);
      signal?.addEventListener("abort", onAbort, { once: true });
      if (signal?.aborted) onAbort();
    });
  }
}

export async function validateFiles(files: File[]): Promise<void> {
  if (files.length !== 25)
    throw new Error(
      `Choose exactly 25 fragments. You selected ${files.length}.`,
    );
  for (const file of files) {
    if (!/\.(png|jpe?g)$/i.test(file.name))
      throw new Error(`${file.name}: choose a PNG or JPEG image.`);
    if (file.size > 256 * 1024)
      throw new Error(
        `${file.name}: each fragment must be 256 KiB or smaller.`,
      );
    let bitmap: ImageBitmap;
    try {
      bitmap = await createImageBitmap(file);
    } catch {
      throw new Error(`${file.name}: this image could not be read.`);
    }
    const validSize = bitmap.width === 64 && bitmap.height === 64;
    bitmap.close();
    if (!validSize)
      throw new Error(`${file.name}: fragments must be 64 × 64 pixels.`);
  }
  // The API additionally checks actual file format, RGB mode and frame count.
}
