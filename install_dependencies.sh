echo "Updating the list of available packages "
sudo apt update
echo "Installing gcc and cmake"
sudo apt install build-essential
sudo apt install cmake
echo "Installing curl"
sudo apt install curl
echo "Installing cargo and rust"
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
echo "Installing additional dependencies needed for cargo to compile the project"
sudo apt install pkg-config
sudo apt install libfontconfig1-dev
echo "Installing ffmeg"
sudo apt install ffmpeg