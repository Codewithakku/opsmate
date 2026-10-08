# Dockerfile for Frontend (Next.js)

# -------------------------
# Build stage
# -------------------------
FROM node:22-alpine AS builder

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
# npm install is used instead of npm ci for flexibility
RUN npm install

# Copy source code
COPY . .

# Accept build args for Next.js browser env variables
ARG NEXT_PUBLIC_WS_URL
ARG NEXT_PUBLIC_API_URL

# Expose them as ENV so Next.js build can see them
ENV NEXT_PUBLIC_WS_URL=$NEXT_PUBLIC_WS_URL
ENV NEXT_PUBLIC_API_URL=$NEXT_PUBLIC_API_URL

# Build the Next.js application
RUN npm run build

# Remove dev-only dependencies
RUN npm prune --production

# -------------------------
# Production stage
# -------------------------
FROM node:22-alpine AS runner

WORKDIR /app

# Production environment
ENV NODE_ENV=production

# Copy built Next.js application
COPY --from=builder /app/.next ./.next

# Copy public folder
COPY --from=builder /app/public ./public

# Copy production node_modules
COPY --from=builder /app/node_modules ./node_modules

# Copy package.json
COPY --from=builder /app/package.json ./package.json

# Next.js port (internal only — Caddy publishes host port 3001)
EXPOSE 5173

# Start Next.js (package.json "start" = next start -p 5173)
CMD ["npm", "run", "start"]