# Use Debian-based image for SQLite compatibility
FROM debian:bullseye-slim  

# Create a non-root user and set correct directory permissions
RUN useradd -m nonroot && mkdir -p /data && chown -R nonroot:nonroot /data

# Set working directory
WORKDIR /app

# Install SQLite, OpenGL for OpenCV, tmux, and cleanup
RUN apt-get update && apt-get install -y --no-install-recommends \
    sqlite3 libgl1-mesa-glx libglib2.0-0 ca-certificates tmux && \
    rm -rf /var/lib/apt/lists/*

# Copy executables
COPY dist/createtable /app/createtable
COPY dist/insertUsers /app/insertUsers
COPY dist/fastapi_app /app/fastapi_app

# Ensure executables are runnable
RUN chmod +x /app/createtable /app/insertUsers /app/fastapi_app

# Copy cron services
COPY sync_server_tocloud/edge_client/new_edge_sync/dist/cron_populate /app/cron_populate
COPY sync_server_tocloud/edge_client/new_edge_sync/dist/cron_pushtocloud /app/cron_pushtocloud
COPY sync_server_tocloud/edge_client/new_edge_sync/dist/cron_scheduler /app/cron_scheduler

# Ensure cron services are executable
RUN chmod +x /app/cron_populate /app/cron_pushtocloud /app/cron_scheduler

# Expose the application port
EXPOSE 9001

# Switch to non-root user
USER nonroot

# Run services in separate tmux panes
CMD ["/bin/sh", "-c", "chown -R nonroot:nonroot /data && \
    /app/cron_populate > /app/cron_populate.log 2>&1 & \
    /app/cron_pushtocloud > /app/cron_pushtocloud.log 2>&1 & \
    /app/cron_scheduler > /app/cron_scheduler.log 2>&1 & \
    /app/createtable && /app/insertUsers && /app/fastapi_app & \
    tail -f /app/cron_populate.log /app/cron_pushtocloud.log /app/cron_scheduler.log"]

