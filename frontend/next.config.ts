import type { NextConfig } from "next";

const backend = (process.env.JIGSAW_API_URL ?? "http://127.0.0.1:8000").replace(
  /\/$/,
  "",
);
const config: NextConfig = {
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${backend}/:path*` }];
  },
};
export default config;
