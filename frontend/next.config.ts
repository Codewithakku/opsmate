import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  reactStrictMode: true,

  turbopack: {},

  allowedDevOrigins: [
    "localhost:3001",
    "127.0.0.1:3001",
    "192.168.0.166:3001",
    "192.168.0.166:5173",
  ],

  async rewrites() {
    // Local-dev bridge so the browser only ever needs the frontend origin:
    // /_backend/* -> Express backend. (In Docker, Caddy routes /api/* instead.)
    return [
      {
        source: "/_backend/:path*",
        destination: `${process.env.BACKEND_URL || "http://localhost:5000"}/:path*`,
      },
    ];
  },

};

export default nextConfig;