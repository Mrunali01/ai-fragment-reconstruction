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
  arrangement: {
    row: number;
    column: number;
    upload_index: number;
    filename: string;
  }[];
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
