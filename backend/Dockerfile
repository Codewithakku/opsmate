# Dockerfile for Backend (Express + WebSocket)
# Use lightweight Node image
FROM node:22-alpine AS builder

WORKDIR /app

# Copy package.json and lock files
COPY package*.json ./

# Install dependencies (including dev for building, but none needed)
RUN npm ci --omit=dev

# Copy source code
COPY . .

# Production stage
FROM node:22-alpine AS runner
WORKDIR /app

# Copy only needed files from builder
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/server.js ./server.js
COPY --from=builder /app/package.json ./package.json


# Expose backend port
EXPOSE 5000

# Start the Express server
CMD ["node", "server.js"]
