import type { NextConfig } from "next";
import createNextIntlPlugin from "next-intl/plugin";

const withNextIntl = createNextIntlPlugin();

const nextConfig: NextConfig = {
  // Copy ONNX Runtime WASM files to static assets
  webpack: (config, { isServer }) => {
    if (!isServer) {
      config.resolve.fallback = {
        ...config.resolve.fallback,
        fs: false,
        path: false,
      };
    }
    return config;
  },
  // Allow WASM files to be served
  headers: async () => [
    {
      source: "/:path*.wasm",
      headers: [
        { key: "Content-Type", value: "application/wasm" },
      ],
    },
  ],
};

export default withNextIntl(nextConfig);
