# Use Debian-based image for SQLite compatibility
FROM debian:latest  

# Create a non-root user and set correct directory permissions
RUN useradd -m nonroot && mkdir -p /data && chown -R nonroot:nonroot /data

# Set working directory
WORKDIR /app

# Install SQLite and OpenGL for OpenCV
RUN apt-get update && apt-get install -y sqlite3 libgl1-mesa-glx libglib2.0-0

# Copy executables
COPY dist/createtable /app/createtable
COPY dist/insertUsers /app/insertUsers
COPY dist/fastapi_app /app/fastapi_app

# Ensure executables are runnable
RUN chmod +x /app/createtable /app/insertUsers /app/fastapi_app

# Install websocket dependencies
# RUN pip install uvicorn[standard]

# Switch to non-root user
USER nonroot

# Expose the application port
EXPOSE 8000

# Run scripts before starting the app
CMD ["/bin/sh", "-c", "chown -R nonroot:nonroot /data && /app/createtable && /app/insertUsers && /app/fastapi_app"]
