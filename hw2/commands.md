1. docker run -it \
   --rm \
   -v ollama:/root/.ollama \
   -p 11434:11434 \
   --name ollama \
   ollama/ollama

2. I use Rancher under Windows 11 + WSL.
   {
   "CreatedAt" : "2024-06-30T11:15:54+02:00",
   "Driver" : "local",
   "Labels" : {
   "com.docker.compose.project" : "llm_zoomcamp_hw2",
   "com.docker.compose.version" : "2.5.1",
   "com.docker.compose.volume" : "ollama"
   },
   "Mountpoint" : "/var/lib/docker/volumes/llm_zoomcamp_hw2_ollama/_data",
   "Name" : "llm_zoomcamp_hw2_ollama",
   "Options" : null,
   "Scope" : "local"
   }