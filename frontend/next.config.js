/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  images: {
    domains: ['asl-api.onrender.com', 'asl-translate-backend.onrender.com'],
  },
}

module.exports = nextConfig 