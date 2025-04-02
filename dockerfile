# Use a smaller Debian-based image for SQLite compatibility
FROM debian:bullseye-slim  

# Create a non-root user and set correct directory permissions
RUN useradd -m nonroot && mkdir -p /data /app && chown -R nonroot:nonroot /data /app

# Set working directory
WORKDIR /app

# Install required packages efficiently
RUN apt-get update && apt-get install -y --no-install-recommends \
    sqlite3 libgl1-mesa-glx libglib2.0-0 ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Copy executables and ensure they are executable in one step
COPY --chmod=0755 dist/createtable dist/insertUsers dist/fastapi_app /app/

# Copy cron services and set permissions in one step
COPY --chmod=0755 \
    sync_server_tocloud/edge_client/new_edge_sync/dist/cron_populate \
    sync_server_tocloud/edge_client/new_edge_sync/dist/cron_pushtocloud \
    sync_server_tocloud/edge_client/new_edge_sync/dist/cron_scheduler /app/

# Create empty log files with correct permissions before switching users
RUN touch /app/cron_populate.log /app/cron_pushtocloud.log /app/cron_scheduler.log && \
    chmod 666 /app/cron_populate.log /app/cron_pushtocloud.log /app/cron_scheduler.log && \
    chown nonroot:nonroot /app/cron_populate.log /app/cron_pushtocloud.log /app/cron_scheduler.log


# Expose the application port
EXPOSE 9001

# Switch to non-root user
USER nonroot

# Run services and keep logs accessible
CMD ["/bin/sh", "-c", " \
    /app/cron_scheduler > /app/cron_scheduler.log 2>&1 & \
    sleep 5; /app/cron_populate > /app/cron_populate.log 2>&1 & \
    sleep 5; /app/cron_pushtocloud > /app/cron_pushtocloud.log 2>&1 & \
    /app/createtable && /app/insertUsers && /app/fastapi_app & \
    tail -f /app/cron_populate.log /app/cron_pushtocloud.log /app/cron_scheduler.log"]
