clear

echo "Building Docker image..."
docker build -t roadvision:1.00 . && echo "Build succeeded." || { echo "Build failed."; exit 1; }

echo "Listing images..."
docker image ls

echo "Removing dangling images..."
docker image rm -f $(docker images -f dangling=true -q) || echo "No dangling images to remove."

echo "Running Docker container..."
docker run -p 80:80 --cpus="4.5" --memory="4g" roadvision:1.00